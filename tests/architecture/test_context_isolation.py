"""A consumer of a context's published language loads none of that context's interior.

import-linter proves it statically; this test proves it at run time, in a fresh interpreter,
for the published package itself and for a module standing in for a downstream context.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIOR = (
    "emblema.catalog.domain",
    "emblema.catalog.ports",
    "emblema.catalog.application",
    "emblema.catalog.adapters",
)


def loaded_modules_after_importing(module: str) -> list[str]:
    script = (
        f"import sys, {module}; "
        "print('\\n'.join(sorted(name for name in sys.modules if name.startswith('emblema.'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


@pytest.mark.parametrize(
    "module",
    ["emblema.catalog.contracts.events", "tests.contracts.downstream"],
)
def test_importing_the_published_language_loads_no_catalog_interior(module: str) -> None:
    loaded = loaded_modules_after_importing(module)

    assert "emblema.catalog.contracts.corpus_version_ref" in loaded
    assert not [name for name in loaded if name.startswith(INTERIOR)]
