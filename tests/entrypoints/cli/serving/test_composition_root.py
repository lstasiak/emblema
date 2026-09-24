"""The process that promotes and withdraws, assembled from what the settings name.

Its one peculiarity is what it must not load: the stack a backbone is trained or adapted with.
Promotion decides on a projection and a checksum, so a process that loaded torch to do it would
carry into the serving image the weight that image exists to leave out.
"""

import subprocess
import sys

import pytest

from emblema.entrypoints.cli.serving.composition_root import CompositionRoot
from emblema.serving.adapters.persistence.promotable_artifact_repository import (
    SqlAlchemyPromotableArtifactRepository,
)
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from tests.support.settings import unreachable_store


def test_without_overrides_the_process_runs_on_what_the_settings_name() -> None:
    root = CompositionRoot(unreachable_store())

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.promotables, SqlAlchemyPromotableArtifactRepository)
    assert isinstance(root.adapters.served, SqlAlchemyServedModelRepository)


def test_a_process_bringing_neither_settings_nor_its_registries_is_refused() -> None:
    with pytest.raises(ValueError, match="registries"):
        CompositionRoot()


def test_assembling_this_process_loads_neither_stack_a_candidate_is_fitted_with() -> None:
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import emblema.entrypoints.cli.serving.composition_root  # noqa: F401\n"
            "print(sorted(m for m in sys.modules if m in {'torch', 'xgboost', 'sklearn'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert read.stdout.strip() == "[]"
