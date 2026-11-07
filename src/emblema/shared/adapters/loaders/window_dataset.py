from collections.abc import Sequence

from torch.utils.data import Dataset


class WindowDataset[Item](Dataset[Item]):
    """Windows addressed by position, which is all a map-style loader asks of a dataset.

    Windows are tokenised once and read many times, never tokenised per epoch: cutting a corpus
    into windows costs minutes of pure Python, and an epoch that pays it again pays it for nothing.
    Any ``Sequence`` of windows will do, so the memory-mapped store of the published corpus slots
    in here unchanged once it exists, as does a list built in a test; a window is a token window
    or whatever else its loader's collation lays out, a window joined with those before it.
    """

    def __init__(self, windows: Sequence[Item]) -> None:
        """Hold ``windows`` in the order given.

        Args:
            windows: The windows of the corpus, in a fixed order; at least one.

        Raises:
            ValueError: If no window is given.
        """
        if not windows:
            raise ValueError("a dataset needs at least one window")
        self._windows = windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Item:
        return self._windows[index]
