"""Where a downloaded corpus's root sits on this machine, for the reports that read one.

``fetch_corpora.py`` unpacks each archive into ``data/raw/<corpus>/`` as the publisher shaped it,
so the directory a reader has to be handed is nested differently per corpus. The markers below say
what that directory must contain; a corpus absent from the table keeps the C-MAPSS shape, a
directory of ``*.txt`` files. Stated once, because two reports need the same answer and would
otherwise disagree about it.
"""

from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

MARKERS = {
    "skab": "data/valve1",
    "smd": "train/machine-1-1.txt",
    "esa_ad": "ESA-Mission1",
    "physionet2012": "set-a",
}


def raw_root(corpus: str) -> Path | None:
    """The directory the corpus's reader is bound to; ``None`` if it was never downloaded."""
    marker = MARKERS.get(corpus, "*.txt")
    hits = sorted((RAW / corpus).rglob(marker)) if (RAW / corpus).is_dir() else []
    return hits[0].parent if hits else None
