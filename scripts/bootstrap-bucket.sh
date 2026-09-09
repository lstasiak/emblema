#!/usr/bin/env bash
# Prepare one S3-compatible bucket for the artifact store: make sure it exists and install the
# lifecycle rules that keep it inside a free-tier quota. Idempotent. The same script runs against
# the local stack (compose service `bootstrap`) and against the remote bucket (same service with
# the remote environment file); only the environment differs.
#
# Environment: AWS_* credentials and AWS_ENDPOINT_URL as the AWS CLI reads them; BUCKET;
# ENVIRONMENT_PREFIXES (space-separated top-level prefixes, e.g. "dev test");
# TRANSIENT_RETENTION_DAYS (age after which `<prefix>/transient/` objects expire).
set -euo pipefail

: "${BUCKET:?BUCKET is required}"
: "${ENVIRONMENT_PREFIXES:?ENVIRONMENT_PREFIXES is required}"
: "${TRANSIENT_RETENTION_DAYS:?TRANSIENT_RETENTION_DAYS is required}"

# Prints present, absent, forbidden or unreachable. HeadBucket is the one probe every provider
# allows a bucket-scoped key to make. A rejected key deserves its own answer: retrying it only
# delays the report and blames the network for a right the token does not carry.
bucket_state() {
  local output
  if output=$(aws s3api head-bucket --bucket "$BUCKET" 2>&1); then
    echo present
  elif grep "404" <<<"$output" >/dev/null; then
    echo absent
  elif grep -E "403|AccessDenied|Forbidden" <<<"$output" >/dev/null; then
    echo forbidden
  else
    echo unreachable
  fi
}

state=unreachable
for _ in $(seq 1 30); do
  state=$(bucket_state)
  [[ $state != unreachable ]] && break
  sleep 2
done

case $state in
  present) echo "bucket ${BUCKET}: present" ;;
  absent)
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
    echo "bucket ${BUCKET}: created"
    ;;
  forbidden)
    echo "bucket ${BUCKET}: access denied. The key needs bucket-level rights, both to read the" \
      "bucket and to write its lifecycle configuration" >&2
    exit 1
    ;;
  *)
    echo "bucket ${BUCKET}: endpoint ${AWS_ENDPOINT_URL:-<default>} unreachable" >&2
    exit 1
    ;;
esac

# One expiry rule per environment prefix; prefixes must sit inside Filter, the rule-level Prefix
# is deprecated and some providers reject it. Abandoned multipart uploads are invisible to
# listings but count against the quota, so they are aborted after a day.
rules=""
for prefix in $ENVIRONMENT_PREFIXES; do
  rules+=$(printf '{"ID": "expire-transient-%s", "Status": "Enabled", "Filter": {"Prefix": "%s/transient/"}, "Expiration": {"Days": %d}},' \
    "$prefix" "$prefix" "$TRANSIENT_RETENTION_DAYS")
done
rules+='{"ID": "abort-incomplete-multipart", "Status": "Enabled", "Filter": {"Prefix": ""}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}'

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET" \
  --lifecycle-configuration "{\"Rules\": [${rules}]}"
echo "bucket ${BUCKET}: lifecycle rules installed"
aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET"
