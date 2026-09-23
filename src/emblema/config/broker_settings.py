from urllib.parse import quote

from pydantic import BaseModel, Field, SecretStr
from pydantic_core import Url


class BrokerSettings(BaseModel):
    """Connection to the broker background work passes through, read as ``EMBLEMA_BROKER__<FIELD>``.

    Nothing has a default: a process that names no broker must fail on startup rather than reach
    for whichever one a default happened to describe.
    """

    host: str
    port: int
    user: str
    password: SecretStr
    vhost: str = Field(
        description="Virtual host the queues live in; '/' is the broker's own default."
    )

    def url(self) -> str:
        """The connection URL, with every part of it percent-encoded before it is placed in it.

        Encoded here rather than left to the URL builder, which refuses a credential holding a
        delimiter it would have to escape and passes an existing ``%`` through unchanged. The
        virtual host needs it as much as the credentials do: the broker's default one is named
        ``/``, which is a path separator until it is encoded.
        """
        return str(
            Url.build(
                scheme="amqp",
                host=self.host,
                port=self.port,
                username=quote(self.user, safe=""),
                password=quote(self.password.get_secret_value(), safe=""),
                path=quote(self.vhost, safe=""),
            )
        )
