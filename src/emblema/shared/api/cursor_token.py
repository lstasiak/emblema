import base64
import binascii
import json

from emblema.shared.kernel.exceptions import InvalidCursorError
from emblema.shared.kernel.paging.cursor import Cursor


class CursorToken:
    """A cursor as it travels to a client and back: one opaque token.

    The parts of the key are written as a JSON list under a URL-safe encoding without padding,
    so the token survives a query string and a JSON field unchanged and a client has no reason
    to read it. A token that does not decode to a list of text parts is refused rather than read
    as an empty cursor.
    """

    @staticmethod
    def encode(cursor: Cursor) -> str:
        document = json.dumps(list(cursor.parts), separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(document).decode("ascii").rstrip("=")

    @staticmethod
    def decode(token: str) -> Cursor:
        """The cursor a client handed back.

        Raises:
            InvalidCursorError: If the token is not one this issued.
        """
        if not token or token != token.strip():
            raise InvalidCursorError("a cursor is non-blank without surrounding whitespace")
        padded = token + "=" * (-len(token) % 4)
        try:
            decoded = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidCursorError(f"not a cursor: {token!r}") from error
        if not isinstance(decoded, list) or not all(isinstance(part, str) for part in decoded):
            raise InvalidCursorError(f"not a cursor: {token!r}")
        try:
            return Cursor(tuple(decoded))
        except InvalidCursorError as error:
            raise InvalidCursorError(f"not a cursor: {token!r}") from error
