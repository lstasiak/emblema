import pytest

from emblema.shared.kernel.tokens import Token, TokenWindow

pytest.importorskip("torch")

from emblema.shared.adapters.loaders.window_dataset import WindowDataset

pytestmark = pytest.mark.ml

WINDOWS = [
    TokenWindow.of([Token(1, 0.25, 0.5, 0.5)]),
    TokenWindow.of([Token(2, -1.0, 0.75, 0.75)]),
]


def test_a_dataset_reports_how_many_windows_it_holds() -> None:
    assert len(WindowDataset(WINDOWS)) == 2


def test_a_window_is_read_back_at_the_position_it_was_given() -> None:
    dataset = WindowDataset(WINDOWS)

    assert [dataset[index] for index in range(len(dataset))] == WINDOWS


def test_a_dataset_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        WindowDataset([])
