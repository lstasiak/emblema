"""Architecture rules are import-linter contracts in pyproject.toml.

The positive test runs them against the real package. The negative tests run the same contracts
against a copy of the package with one deliberate violation injected, so every rule is shown to
break for the reason it exists rather than to pass by absence of code.
"""

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_SOURCE = REPO_ROOT / "src" / "emblema"


def _contract_names_by_id() -> dict[str, str]:
    with PYPROJECT.open("rb") as handle:
        contracts = tomllib.load(handle)["tool"]["importlinter"]["contracts"]
    return {contract["id"]: contract["name"] for contract in contracts}


CONTRACT_NAMES = _contract_names_by_id()


@dataclass(frozen=True)
class Violation:
    """One deliberate breach of one contract.

    Attributes:
        contract_id: Contract expected to break.
        modules: Files to add to a copy of the package, relative to ``emblema/``.
        evidence: Import line the report must name, proving the copy was analysed.
    """

    contract_id: str
    modules: dict[str, str]
    evidence: str


VIOLATIONS = [
    Violation(
        contract_id="context-independence",
        modules={
            "catalog/domain/__init__.py": "",
            "catalog/domain/leak.py": "import emblema.pretraining.domain\n",
            "pretraining/domain/__init__.py": "",
        },
        evidence="emblema.catalog.domain.leak -> emblema.pretraining.domain",
    ),
    Violation(
        contract_id="hexagonal-layers",
        modules={
            "catalog/domain/__init__.py": "",
            "catalog/domain/entity.py": "import emblema.catalog.adapters.persistence\n",
            "catalog/adapters/__init__.py": "",
            "catalog/adapters/persistence.py": "",
        },
        evidence="emblema.catalog.domain.entity -> emblema.catalog.adapters.persistence",
    ),
    Violation(
        contract_id="hexagonal-layers",
        modules={
            "catalog/application/__init__.py": "",
            "catalog/application/use_case.py": "import emblema.catalog.adapters.persistence\n",
            "catalog/adapters/__init__.py": "",
            "catalog/adapters/persistence.py": "",
        },
        evidence="emblema.catalog.application.use_case -> emblema.catalog.adapters.persistence",
    ),
    Violation(
        contract_id="pure-core",
        modules={
            "catalog/domain/__init__.py": "",
            "catalog/domain/model.py": "import torch\n",
        },
        evidence="emblema.catalog.domain.model -> torch",
    ),
    Violation(
        contract_id="pure-core",
        modules={
            "catalog/application/__init__.py": "",
            "catalog/application/dto.py": "from pydantic import BaseModel\n",
        },
        evidence="emblema.catalog.application.dto -> pydantic",
    ),
    Violation(
        contract_id="pure-core",
        modules={"shared/kernel/tensor.py": "import numpy\n"},
        evidence="emblema.shared.kernel.tensor -> numpy",
    ),
    Violation(
        contract_id="config-only-at-the-edges",
        modules={
            "catalog/application/__init__.py": "",
            "catalog/application/use_case.py": "from emblema.config.settings import Settings\n",
        },
        evidence="emblema.catalog.application.use_case -> emblema.config.settings",
    ),
    Violation(
        contract_id="contracts-depend-only-on-shared",
        modules={
            "catalog/contracts/__init__.py": "",
            "catalog/contracts/published.py": "import emblema.catalog.domain\n",
            "catalog/domain/__init__.py": "",
        },
        evidence="emblema.catalog.contracts.published -> emblema.catalog.domain",
    ),
    Violation(
        contract_id="shared-imports-no-context",
        modules={"shared/kernel/leak.py": "import emblema.catalog\n"},
        evidence="emblema.shared.kernel.leak -> emblema.catalog",
    ),
    Violation(
        contract_id="entrypoints-are-outermost",
        modules={
            "catalog/adapters/__init__.py": "",
            "catalog/adapters/cli_glue.py": "import emblema.entrypoints\n",
            "entrypoints/__init__.py": "",
        },
        evidence="emblema.catalog.adapters.cli_glue -> emblema.entrypoints",
    ),
]


def lint_imports(package_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the CLI with the repository contracts, optionally against another copy of the package.

    A copy placed first on PYTHONPATH shadows the installed package, so the unchanged contracts
    are checked against code with injected violations.
    """
    executable = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    assert executable, "lint-imports must be installed next to the test interpreter"
    env = os.environ.copy()
    if package_root is not None:
        env["PYTHONPATH"] = str(package_root)
    return subprocess.run(
        [executable, "--config", str(PYPROJECT), "--no-cache"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def package_copy_with(root: Path, modules: dict[str, str]) -> Path:
    copy = root / "emblema"
    shutil.copytree(PACKAGE_SOURCE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    for relative, source in modules.items():
        target = copy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return root


def test_repository_package_keeps_every_contract() -> None:
    result = lint_imports()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout


def test_every_contract_has_a_violation_test() -> None:
    assert {violation.contract_id for violation in VIOLATIONS} == set(CONTRACT_NAMES)


@pytest.mark.parametrize(
    "violation",
    [pytest.param(v, id=f"{v.contract_id}[{v.evidence}]") for v in VIOLATIONS],
)
def test_violation_breaks_its_contract(violation: Violation, tmp_path: Path) -> None:
    result = lint_imports(package_copy_with(tmp_path, violation.modules))

    assert result.returncode != 0, result.stdout
    assert f"{CONTRACT_NAMES[violation.contract_id]} BROKEN" in result.stdout, result.stdout
    assert violation.evidence in result.stdout, result.stdout


def test_context_may_import_published_contracts_of_another_context(tmp_path: Path) -> None:
    modules = {
        "catalog/application/__init__.py": "",
        "catalog/application/use_case.py": "import emblema.pretraining.contracts.published\n",
        "pretraining/contracts/__init__.py": "",
        "pretraining/contracts/published.py": "",
    }

    result = lint_imports(package_copy_with(tmp_path, modules))

    assert result.returncode == 0, result.stdout + result.stderr
