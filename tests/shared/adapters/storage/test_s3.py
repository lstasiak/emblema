"""Behaviour of the S3 adapter that needs no bucket: client configuration and error mapping."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

REF = content_address(b"weights", Retention.DURABLE)


class Stubbed(NamedTuple):
    store: S3ArtifactStore
    responses: Stubber


@pytest.fixture
def stubbed() -> Iterator[Stubbed]:
    client: S3Client = boto3.client(
        "s3", region_name="garage", aws_access_key_id="key", aws_secret_access_key="secret"
    )
    with Stubber(client) as responses:
        yield Stubbed(S3ArtifactStore(client, "emblema", "test"), responses)
        responses.assert_no_pending_responses()


def test_connect_uses_path_style_and_sends_checksums_only_when_required() -> None:
    store = S3ArtifactStore.connect(
        endpoint_url="http://127.0.0.1:3900",
        region="garage",
        access_key="key",
        secret_key="secret",
        bucket="emblema",
        key_prefix="dev",
    )

    # botocore's Config declares its options dynamically; the stubs know none of them.
    config: Any = store._client.meta.config
    assert config.s3 == {"addressing_style": "path"}
    assert config.request_checksum_calculation == "when_required"
    assert config.response_checksum_validation == "when_required"


@pytest.mark.parametrize("prefix", ["", " ", "/dev", "dev/", " dev"])
def test_rejects_blank_or_slashed_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="key prefix"):
        S3ArtifactStore.connect(
            endpoint_url="http://127.0.0.1:3900",
            region="garage",
            access_key="key",
            secret_key="secret",
            bucket="emblema",
            key_prefix=prefix,
        )


def test_objects_live_under_the_environment_prefix(stubbed: Stubbed) -> None:
    stubbed.responses.add_response(
        "head_object", {}, {"Bucket": "emblema", "Key": f"test/{REF.key}"}
    )

    assert stubbed.store.exists(REF)


def test_missing_object_on_get_is_reported_as_not_found(stubbed: Stubbed) -> None:
    stubbed.responses.add_client_error(
        "get_object", service_error_code="NoSuchKey", http_status_code=404
    )

    with pytest.raises(ArtifactNotFoundError):
        stubbed.store.get(REF)


def test_other_service_errors_propagate_unchanged(stubbed: Stubbed) -> None:
    stubbed.responses.add_client_error(
        "head_object", service_error_code="AccessDenied", http_status_code=403
    )

    with pytest.raises(ClientError):
        stubbed.store.exists(REF)


def test_a_read_refused_by_the_service_is_not_reported_as_a_missing_artifact(
    stubbed: Stubbed,
) -> None:
    # A bucket that refuses us and a key that is not there are different facts, and only the
    # second one means the artifact does not exist.
    stubbed.responses.add_client_error(
        "get_object", service_error_code="AccessDenied", http_status_code=403
    )

    with pytest.raises(ClientError):
        stubbed.store.get(REF)
