class UnjoinableWindowsError(ValueError):
    """Windows that cannot be joined with those before them: unplaced, uneven or static."""


class OverlappingJoinedWindowsError(UnjoinableWindowsError):
    """Two windows of one axis overlap in time, so neither is the other's history."""
