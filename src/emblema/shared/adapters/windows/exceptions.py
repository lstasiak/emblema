"""Failures of the window block codec, on either side of the file."""


class WindowBlockError(Exception):
    pass


class MalformedBlockError(WindowBlockError, ValueError):
    pass


class BlockClosedError(WindowBlockError):
    pass


class HeaderOverflowError(WindowBlockError):
    pass


class UnstorableWindowError(WindowBlockError, ValueError):
    """A window that, stored at the block's precision, would no longer be a valid window."""
