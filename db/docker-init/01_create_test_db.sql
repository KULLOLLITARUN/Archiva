-- A separate database for the Postgres-backed tests. They TRUNCATE every
-- table they touch and refuse to run unless the database name ends in _test
-- (tests/conftest.py), so they can never wipe the real `archiva` database.
-- Tables are created by the app/tests themselves (db/schema.sql via init_db()).
CREATE DATABASE archiva_test OWNER archiva;
