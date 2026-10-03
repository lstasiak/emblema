from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class NamedIssuer:
    """An issuer named whole: who signs, for whom, and where its keys are published.

    A dataclass rather than a model: never parsed or serialised, only handed from the settings
    to the composition root.
    """

    issuer: str
    audience: str
    jwks_url: str
