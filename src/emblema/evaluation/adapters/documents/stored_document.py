"""What every JSON document this context hands to another machine is read through.

A document says what it is and at which version, and a reader refuses anything else before it
reads a field: an order read as a result, or a result of a format this code no longer reads,
would otherwise fail on whichever field happened to differ first.
"""

import json
from typing import Any

from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError

# What a document that is not one of ours fails with on the way in, whichever part broke.
MALFORMED = (KeyError, TypeError, ValueError, AttributeError)


def read_document(content: bytes, format_: str, version: int, what: str) -> dict[str, Any]:
    """The JSON object those bytes hold, refused unless it says it is ``format_`` at ``version``.

    Raises:
        UnreadableCampaignDocumentError: If the bytes are not JSON, not an object, or another
            document.
    """
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UnreadableCampaignDocumentError(f"{what} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise UnreadableCampaignDocumentError(f"{what} is not an object")
    if document.get("format") != format_ or document.get("version") != version:
        raise UnreadableCampaignDocumentError(
            f"document is a {document.get('format')!r} of version {document.get('version')!r}, "
            f"not the {what} this reads"
        )
    return document
