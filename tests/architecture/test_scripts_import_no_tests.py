"""Developer scripts depend on the package and on each other, never on the test suite.

A test may import the script it tests. A script that imported the tests would make the two depend
on each other, and a change to a fixture could quietly change what a report measures.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def imported_packages(source: str) -> set[str]:
    """The top-level packages ``source`` imports by absolute import, in either form."""
    packages = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            packages |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            packages.add(node.module.split(".")[0])
    return packages


@pytest.mark.parametrize("script", sorted(SCRIPTS.rglob("*.py")), ids=lambda path: path.name)
def test_a_script_imports_nothing_from_the_tests(script: Path) -> None:
    assert "tests" not in imported_packages(script.read_text(encoding="utf-8"))


def test_an_import_of_the_tests_is_recognised_in_either_form() -> None:
    assert "tests" in imported_packages("from tests.support.settings import unreachable_store\n")
    assert "tests" in imported_packages("import tests.support.token_tensors as tensors\n")
    assert "tests" in imported_packages("def f():\n    from tests import support\n")
    assert "tests" not in imported_packages("from scripts.reporting import table\n")
    assert "tests" not in imported_packages("from .tests import helper\n")
