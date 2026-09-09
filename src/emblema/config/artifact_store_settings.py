from pydantic import BaseModel, Field, SecretStr


class ArtifactStoreSettings(BaseModel):
    """Connection to the S3-compatible bucket holding artifacts.

    A nested plain model rather than a settings class of its own: the parent reads
    ``EMBLEMA_ARTIFACT_STORE__<FIELD>`` through one delimiter, so there is a single place where
    the environment is parsed and a single precedence order. Nothing describing a bucket has a
    default value: a process that names no store must fail on startup rather than write into
    whichever one the defaults happened to describe. Local values live in ``env.example``.
    """

    endpoint_url: str = Field(
        description="S3 endpoint of the service holding the bucket, scheme included.",
    )
    region: str = Field(
        description="Signing region: 'garage' for the local stack, 'auto' for Cloudflare R2.",
    )
    bucket: str = Field(description="Bucket the artifacts live in.")
    key_prefix: str = Field(
        description=(
            "Top-level prefix isolating this environment's objects inside the bucket. Usually the "
            "environment name, but a run that has to clean up after itself may nest deeper."
        ),
    )
    access_key: SecretStr | None = Field(
        default=None, description="Access key id. None defers to the SDK credential chain."
    )
    secret_key: SecretStr | None = Field(
        default=None, description="Secret access key. None defers to the SDK credential chain."
    )
