"""The ports package of a bounded context holds interfaces and nothing else.

A port is a ``Protocol``, one per module; the types it speaks are the domain's and the failures
it reports are domain exceptions, so anything else under ``ports`` is misplaced.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from types import ModuleType

import pytest

# ``typing.is_protocol`` arrived in Python 3.13; the backport serves the 3.12 floor.
from typing_extensions import is_protocol

CONTEXTS = ("catalog", "pretraining", "evaluation", "serving")


def port_modules() -> list[ModuleType]:
    modules = []
    for context in CONTEXTS:
        try:
            package = importlib.import_module(f"emblema.{context}.ports")
        except ModuleNotFoundError:
            continue
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            modules.append(importlib.import_module(info.name))
    return modules


MODULES = port_modules()


def defined_in(module: ModuleType, predicate: Callable[[object], bool]) -> list[str]:
    return [
        name
        for name, member in inspect.getmembers(module, predicate)
        if getattr(member, "__module__", None) == module.__name__ and not name.startswith("_")
    ]


def test_some_context_has_ports() -> None:
    assert MODULES


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_a_ports_module_defines_exactly_one_protocol(module: ModuleType) -> None:
    classes = defined_in(module, inspect.isclass)

    assert len(classes) == 1, classes
    assert is_protocol(getattr(module, classes[0])), classes


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_a_ports_module_defines_no_function(module: ModuleType) -> None:
    assert defined_in(module, inspect.isfunction) == []
