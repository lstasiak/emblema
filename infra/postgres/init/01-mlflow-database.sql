-- MLflow keeps its own tables in its own database: raw training logs never share a schema with
-- the domain facts the application stores.
CREATE DATABASE mlflow;
