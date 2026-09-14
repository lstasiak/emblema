# S3 adapter against the remote bucket (Cloudflare R2)

CI has no credentials for the remote bucket. The proof that the store is switched
by configuration alone is running the unchanged contract suite, and the unchanged bootstrap
script, against R2 from the development machine.

## Procedure

`.env.r2` holds the R2 endpoint, `region=auto`, bucket, key prefix `test` and an API token for that
bucket (see the commented block in `env.example`). Never commit it. Every `EMBLEMA_` variable has to
be in the file: one left out falls back to `.env`, and the run would exercise the local stack while
claiming to have reached R2 — hence the first command below, whose output belongs in the record.

Writing lifecycle configuration is a bucket-level operation, so the token needs bucket-level rights,
not only object read/write. With an object-scoped token the script now stops at once and says so
instead of retrying the endpoint.

```sh
uv run --env-file .env.r2 python -c "from emblema.config.settings import Settings; s = Settings().artifact_store; print(s.endpoint_url, s.bucket, s.key_prefix)"
docker compose --env-file .env.r2 run --rm --no-deps bootstrap
uv run --env-file .env.r2 pytest -m integration tests/shared/ports/test_artifact_store_contract.py -v
```

Record: date, boto3 version, the endpoint/bucket/prefix line above, R2 account region hint, bootstrap
output (lifecycle rules as R2 echoes them back), pytest summary, and whether the test prefix was
empty afterwards (the fixture deletes what it wrote; whatever a killed run leaves behind sits under
`<prefix>/transient/`, which the lifecycle rule sweeps).

Points to settle:

- R2 accepts the lifecycle document (`Filter.Prefix` rules with `Expiration.Days`, empty-prefix
  `AbortIncompleteMultipartUpload`).
- `request_checksum_calculation="when_required"` is sufficient; no `x-amz-checksum` rejections.
- Class A/B operation counts of one suite run, to size the free tier against future CI use.

## Runs

### 2026-09-09 — MacBook M1 against Cloudflare R2

Result: pass. The unchanged contract suite ran against the remote bucket with
`uv run --env-file .env.r2 pytest -m integration`, and `scripts/bootstrap-bucket.sh` installed the
lifecycle rules on R2 through the same compose service that targets Garage. Nothing but the
environment file differed, which is the evidence for the DoD criterion.

Settled here:

- R2 accepts the lifecycle document, prefix filters and the empty-prefix abort-multipart rule.
- `request_checksum_calculation="when_required"` is sufficient; no checksum trailer was rejected.
- The R2 API token must carry bucket-level rights: writing a lifecycle configuration is a bucket
  operation, and an object-scoped token is refused.
- The S3 endpoint is the account URL without the bucket name; the dashboard displays it with the
  bucket appended, which path-style addressing would then duplicate.

Not captured in this record: date-stamped boto3 version, the endpoint/bucket/prefix line printed
before the run, the bootstrap output as R2 echoed it, the pytest summary, and confirmation that the
test prefix was empty afterwards. Append them from the terminal that ran it.
