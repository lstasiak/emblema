from pydantic import BaseModel, SecretStr
from sqlalchemy import URL


class DatabaseSettings(BaseModel):
    """Connection to the metadata database, read as ``EMBLEMA_DATABASE__<FIELD>``.

    Nothing has a default: a process that names no database must fail on startup rather than
    reach for whichever one a default happened to describe.
    """

    host: str
    port: int
    name: str
    user: str
    password: SecretStr

    def sqlalchemy_url(self) -> URL:
        """The connection URL, credentials escaped by construction rather than by formatting."""
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.name,
        )
