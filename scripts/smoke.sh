#!/usr/bin/env bash
# Smoke test of the local stack after `docker compose up -d --wait`: each service does the one
# thing the platform needs from it. Stops at the first failure. Needs curl and docker compose.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MLFLOW_URL=${MLFLOW_URL:-http://localhost:${MLFLOW_PORT:-5000}}
BUCKET=${EMBLEMA_ARTIFACT_STORE__BUCKET:-emblema}
PG_USER=${POSTGRES_USER:?POSTGRES_USER is required (copy env.example to .env)}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

psql_scalar() {
  docker compose exec -T postgres psql -U "$PG_USER" -d emblema -tAc "$1"
}

echo "postgres: databases and context schemas"
[[ $(psql_scalar "select count(*) from pg_database where datname = 'mlflow'") == 1 ]] \
  || fail "database mlflow is missing"
schemas=$(psql_scalar "select string_agg(schema_name, ',' order by schema_name) from information_schema.schemata where schema_name in ('catalog', 'evaluation', 'pretraining', 'serving')")
[[ $schemas == "catalog,evaluation,pretraining,serving" ]] || fail "expected four context schemas, got '${schemas}'"

echo "bucket: lifecycle rules"
# `-T` denies the container a terminal, which aws-cli v2 would take as an invitation to page its
# output. Reading each pipe to the end keeps `pipefail` from seeing a writer killed by SIGPIPE.
docker compose run --rm -T --no-deps --entrypoint aws bootstrap \
  s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" \
  | grep "transient" >/dev/null \
  || fail "no lifecycle rule for transient artifacts on ${BUCKET}"

echo "mlflow: a run with a metric and an artifact"
post() {
  curl -fsS -X POST -H "Content-Type: application/json" "${MLFLOW_URL}/api/2.0/mlflow/$1" -d "$2"
}
now_ms() { echo "$(date +%s)000"; }

experiment=$(post experiments/create "{\"name\": \"smoke-$(date +%s)\"}" \
  | grep -oE '"experiment_id": ?"[0-9]+"' | grep -oE '[0-9]+')
run=$(post runs/create "{\"experiment_id\": \"${experiment}\", \"start_time\": $(now_ms)}" \
  | grep -oE '"run_id": ?"[a-f0-9]{32}"' | head -1 | grep -oE '[a-f0-9]{32}')
post runs/log-metric "{\"run_id\": \"${run}\", \"key\": \"smoke\", \"value\": 1.0, \"timestamp\": $(now_ms), \"step\": 0}" >/dev/null
# Artifacts go through the server into the bucket: this is the MLflow -> S3 wiring under test.
curl -fsS -X PUT --data-binary "smoke" \
  "${MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/${experiment}/${run}/artifacts/smoke.txt" >/dev/null
post runs/update "{\"run_id\": \"${run}\", \"status\": \"FINISHED\", \"end_time\": $(now_ms)}" >/dev/null

curl -fsS "${MLFLOW_URL}/api/2.0/mlflow/runs/get?run_id=${run}" \
  | grep -E '"key": ?"smoke"' >/dev/null \
  || fail "metric not recorded on run ${run}"
[[ $(curl -fsS "${MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/${experiment}/${run}/artifacts/smoke.txt") == "smoke" ]] \
  || fail "artifact not readable back from run ${run}"

echo "OK: postgres, bucket and mlflow are ready"
