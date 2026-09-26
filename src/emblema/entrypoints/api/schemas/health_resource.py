from pydantic import BaseModel


class HealthResource(BaseModel):
    """The process is up."""

    status: str
