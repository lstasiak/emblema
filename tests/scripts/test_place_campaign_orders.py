"""Campaigns defined and ordered from a plan; the registry and the store are not touched."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.place_campaign_orders import _answer_of, main, read_plan

COMMIT = "a" * 40
CAMPAIGN = "11111111-1111-1111-1111-111111111111"
DIGEST = "d" * 64
WEIGHTS = "e" * 64


class FakeCli:
    def __init__(self, *, define: str = CAMPAIGN, order: str | None = None) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.define = define
        self.order = order or f"durable/sha256/{DIGEST} sha256:{DIGEST}"

    def __call__(self, weights: str, arguments: Sequence[str]) -> str:
        self.calls.append((weights, tuple(arguments)))
        return self.define if arguments[0] == "define" else self.order


def plan_at(tmp_path: Path) -> list[str]:
    plan = tmp_path / "plan.toml"
    plan.write_text(
        'worker_env = "env.sh"\npool = "ml"\n'
        f'[[campaign]]\nkey = "one"\nfile = "a.toml"\ntask = "t"\nweights = "{WEIGHTS}"\n'
    )
    return [
        "--plan",
        str(plan),
        "--campaigns",
        str(tmp_path / "campaigns.txt"),
        "--orders",
        str(tmp_path / "orders.txt"),
    ]


def test_the_answer_is_the_last_printed_line_whatever_the_tools_said_beside_it() -> None:
    assert _answer_of(0, "noise\nanswer\n", "a warning\n") == "answer"
    assert _answer_of(1, "noise\n", "why it refused\n") == "why it refused"
    assert _answer_of(1, "only here\n", "") == "only here"
    assert _answer_of(0, "", "") == ""


def test_the_settings_are_found_beside_the_plan(tmp_path: Path) -> None:
    arguments = plan_at(tmp_path)

    assert read_plan(Path(arguments[1]))[0] == tmp_path / "env.sh"


def test_a_campaign_is_defined_then_ordered_and_written_as_a_cell_reads_them(
    tmp_path: Path,
) -> None:
    cli = FakeCli()

    main(plan_at(tmp_path), run=cli, state=lambda: (COMMIT, True))

    assert cli.calls == [
        (WEIGHTS, ("define", "--file", "a.toml", "--task", "t")),
        (WEIGHTS, ("order", "--campaign", CAMPAIGN, "--pool", "ml")),
    ]
    assert (tmp_path / "campaigns.txt").read_text() == f"one {CAMPAIGN} {WEIGHTS}\n"
    assert (tmp_path / "orders.txt").read_text() == f"# from {COMMIT}\none {CAMPAIGN} {DIGEST}\n"


def test_run_again_nothing_is_defined_or_ordered_twice(tmp_path: Path) -> None:
    cli = FakeCli()
    main(plan_at(tmp_path), run=cli, state=lambda: (COMMIT, True))
    calls = len(cli.calls)

    main(plan_at(tmp_path), run=cli, state=lambda: (COMMIT, True))

    assert len(cli.calls) == calls


def test_a_refused_order_keeps_the_campaign_so_the_next_run_only_orders(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"\[one\] order refused: no"):
        main(plan_at(tmp_path), run=FakeCli(order="no"), state=lambda: (COMMIT, True))
    assert CAMPAIGN in (tmp_path / "campaigns.txt").read_text()

    cli = FakeCli()
    main(plan_at(tmp_path), run=cli, state=lambda: (COMMIT, True))

    assert [call[1][0] for call in cli.calls] == ["order"]


def test_a_refused_definition_is_named_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"\[one\] define refused: bad"):
        main(plan_at(tmp_path), run=FakeCli(define="bad"), state=lambda: (COMMIT, True))

    assert not (tmp_path / "campaigns.txt").exists()


def test_an_order_whose_key_and_checksum_disagree_is_refused(tmp_path: Path) -> None:
    answer = f"durable/sha256/{DIGEST} sha256:{WEIGHTS}"

    with pytest.raises(SystemExit, match="order refused"):
        main(plan_at(tmp_path), run=FakeCli(order=answer), state=lambda: (COMMIT, True))


def test_a_dirty_tree_is_refused_before_anything_runs(tmp_path: Path) -> None:
    cli = FakeCli()

    with pytest.raises(SystemExit, match="not clean"):
        main(plan_at(tmp_path), run=cli, state=lambda: (COMMIT, False))

    assert cli.calls == []


def test_orders_that_name_no_commit_are_not_added_to(tmp_path: Path) -> None:
    arguments = plan_at(tmp_path)
    (tmp_path / "orders.txt").write_text("")

    with pytest.raises(SystemExit, match="names no commit"):
        main(arguments, run=FakeCli(), state=lambda: (COMMIT, True))


def test_orders_of_another_commit_are_not_added_to(tmp_path: Path) -> None:
    arguments = plan_at(tmp_path)
    (tmp_path / "orders.txt").write_text(f"# from {'b' * 40}\n")

    with pytest.raises(SystemExit, match="were placed at"):
        main(arguments, run=FakeCli(), state=lambda: (COMMIT, True))
