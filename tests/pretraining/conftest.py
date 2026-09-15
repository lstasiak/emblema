import pytest


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A database file is MLflow without a server: the same client and store, no network.

    One database for the whole session: MLflow migrates a fresh one before its first write, and
    that migration costs more than every test that uses it. The directory of files it used to
    offer is in maintenance mode, so this is the local backend it still supports.
    """
    return f"sqlite:///{(tmp_path_factory.mktemp('mlflow') / 'mlflow.db').as_posix()}"
