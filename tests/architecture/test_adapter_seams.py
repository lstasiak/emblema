"""Interfaces live in the ports packages; the few outside them are named adapter seams.

A port is what the application core asks of the outside, typed in the domain's words. An adapter
may also need something it cannot import — a torch module another context owns — and say so with
a ``Protocol`` of its own, typed in its technology and implemented by the process that composes
it. Such a seam is not a port and must not pass for one, so every one is listed here: a new one
is a decision, not a drift.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "emblema"
ADAPTER_SEAMS = {
    "emblema.evaluation.adapters.torch.backbone_factory.BackboneFactory",
    "emblema.shared.adapters.tensors.grown_parameters.GrownParameters",
}


def protocols_in(source: str, module: str) -> set[str]:
    """Qualified names of the classes ``source`` defines with ``Protocol`` among their bases."""
    return {
        f"{module}.{node.name}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and any(names_protocol(base) for base in node.bases)
    }


def names_protocol(base: ast.expr) -> bool:
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Attribute):
        return base.attr == "Protocol"
    return isinstance(base, ast.Name) and base.id == "Protocol"


def module_of(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def protocols_outside_ports() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        module = module_of(path)
        if "ports" not in module.split("."):
            found |= protocols_in(path.read_text(encoding="utf-8"), module)
    return found


def test_every_protocol_outside_the_ports_packages_is_a_named_adapter_seam() -> None:
    assert protocols_outside_ports() == ADAPTER_SEAMS


def test_a_named_seam_belongs_to_an_adapters_package() -> None:
    assert all(".adapters." in seam for seam in ADAPTER_SEAMS)


def test_a_protocol_is_recognised_in_either_spelling() -> None:
    assert protocols_in("class A(Protocol): ...\n", "m") == {"m.A"}
    assert protocols_in("class B(typing.Protocol[T]): ...\n", "m") == {"m.B"}
    assert protocols_in("class C(Base): ...\n", "m") == set()
