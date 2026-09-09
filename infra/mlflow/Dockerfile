# Tracking server. The official MLflow image ships neither a Postgres driver nor an S3 client,
# and a plain Python base builds the same way on arm64 and amd64.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN pip install "mlflow==3.16.0" "psycopg[binary]>=3.2,<4" "boto3>=1.43,<2"

RUN useradd --system --create-home --uid 10001 mlflow
USER mlflow
WORKDIR /home/mlflow

EXPOSE 5000
CMD ["mlflow", "server", "--host=0.0.0.0", "--port=5000"]
