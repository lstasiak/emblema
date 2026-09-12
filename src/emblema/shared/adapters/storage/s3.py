from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Self

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from emblema.shared.adapters.storage.files import CHUNK_SIZE, write_verified
from emblema.shared.adapters.storage.layout import content_address, file_address
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention
from emblema.shared.ports.exceptions import ArtifactIntegrityError, ArtifactNotFoundError

if TYPE_CHECKING:
    from botocore.response import StreamingBody
    from mypy_boto3_s3 import S3Client


class S3ArtifactStore:
    """Artifact store in one S3-compatible bucket.

    The same class serves Garage in the local stack and the remote bucket that GPU platforms
    reach; only the connection differs. ``key_prefix`` isolates the environments that share a
    bucket (``dev``, ``test``) and is not part of the reference, so a reference recorded in one
    environment names the same content in another.
    """

    # Codes an S3-compatible service returns for a missing object: HeadObject carries only the
    # HTTP status, GetObject names the condition.
    _NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

    def __init__(self, client: "S3Client", bucket: str, key_prefix: str) -> None:
        if not key_prefix or key_prefix != key_prefix.strip().strip("/"):
            raise ValueError("key prefix must be non-empty without surrounding slashes or spaces")
        self._client = client
        self._bucket = bucket
        self._key_prefix = key_prefix

    @classmethod
    def connect(
        cls,
        *,
        endpoint_url: str,
        region: str,
        access_key: str | None,
        secret_key: str | None,
        bucket: str,
        key_prefix: str,
    ) -> Self:
        """Open a client with the settings every S3-compatible service we target agrees on.

        Path-style addressing keeps bucket names out of the host name, which a local endpoint
        cannot resolve. Checksums are sent only when an operation requires them: recent SDKs
        default to signing every upload with a CRC trailer, which Cloudflare R2 rejects.
        ``None`` credentials defer to the SDK's own credential chain.
        """
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        return cls(client, bucket, key_prefix)

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        ref = content_address(content, retention)
        if not self.exists(ref):
            self._client.put_object(Bucket=self._bucket, Key=self._object_key(ref), Body=content)
        return ref

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        ref = file_address(source, retention)
        if not self.exists(ref):
            # The file is uploaded by path rather than read here: boto3 splits anything past its
            # threshold into a multipart upload, which is what makes a corpus-sized artifact
            # survive a laptop connection.
            self._client.upload_file(str(source), self._bucket, self._object_key(ref))
        return ref

    def get(self, ref: ArtifactRef) -> bytes:
        content = self._body(ref).read()
        if not ref.checksum.matches(content):
            raise ArtifactIntegrityError(f"content under {ref.key!r} does not match {ref.checksum}")
        return content

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        write_verified(ref, self._streamed(ref), destination)

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._object_key(ref))
        except ClientError as error:
            if self._is_not_found(error):
                return False
            raise
        return True

    def _body(self, ref: ArtifactRef) -> "StreamingBody":
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._object_key(ref))
        except ClientError as error:
            if self._is_not_found(error):
                raise ArtifactNotFoundError(ref.key) from error
            raise
        return response["Body"]

    def _streamed(self, ref: ArtifactRef) -> Iterator[bytes]:
        body = self._body(ref)
        while chunk := body.read(CHUNK_SIZE):
            yield chunk

    def _object_key(self, ref: ArtifactRef) -> str:
        return f"{self._key_prefix}/{ref.key}"

    @classmethod
    def _is_not_found(cls, error: ClientError) -> bool:
        return error.response.get("Error", {}).get("Code") in cls._NOT_FOUND_CODES
