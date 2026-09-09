-- One schema per bounded context in the application database. Tables arrive with the migrations
-- of each context; no foreign key ever crosses a schema boundary.
CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS pretraining;
CREATE SCHEMA IF NOT EXISTS evaluation;
CREATE SCHEMA IF NOT EXISTS serving;
