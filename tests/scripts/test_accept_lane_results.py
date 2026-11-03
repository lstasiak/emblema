"""The block a cell prints, read into results and accepted; the registry is not touched."""

from pathlib import Path

import pytest

from scripts.accept_lane_results import LaneResult, main, read_results

A, B = "a" * 64, "b" * 64


def report(label: str, digest: str, status: int = 0) -> str:
    return f"[{label}] exit {status} in 73 s: durable/sha256/{digest} sha256:{digest}"


def test_the_last_report_of_a_label_wins_and_what_surrounds_the_block_is_ignored() -> None:
    text = "\n".join(
        (
            "=== results (paste back)",
            report("one-0", A, status=1),
            report("two-0", B),
            report("one-0", B),
            "=== failed",
        )
    )

    assert read_results(text) == [
        LaneResult(label="one-0", digest=B),
        LaneResult(label="two-0", digest=B),
    ]


@pytest.mark.parametrize(
    "line",
    [
        report("one-0", A, status=3),
        "[one-0] exit 0 in 5 s: ",
        f"[one-0] exit 0 in 5 s: durable/sha256/{A} sha256:{B}",
    ],
)
def test_a_lane_that_did_not_end_cleanly_has_no_result(line: str) -> None:
    assert read_results(line) == [LaneResult(label="one-0", digest=None)]


def test_every_clean_result_is_accepted_by_its_digest_and_logged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results.txt"
    results.write_text(f"{report('one-0', A)}\n{report('two-0', B)}\n")
    log = tmp_path / "accept.log"
    accepted: list[str] = []

    def accept(digest: str) -> tuple[int, str]:
        accepted.append(digest)
        return 0, "accepted"

    main(["--results", str(results), "--log", str(log)], accept=accept)

    assert accepted == [A, B]
    assert log.read_text() == "[one-0] exit 0: accepted\n[two-0] exit 0: accepted\n"
    assert "accepted: 2 of 2" in capsys.readouterr().out


def test_a_refusal_or_an_unclean_lane_ends_in_failure_after_the_rest_went_through(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.txt"
    results.write_text(f"{report('one-0', A, status=1)}\n{report('two-0', B)}\n")
    log = tmp_path / "accept.log"

    with pytest.raises(SystemExit) as stopped:
        main(["--results", str(results), "--log", str(log)], accept=lambda _: (0, "accepted"))

    assert stopped.value.code == 1
    assert "[one-0] not accepted" in log.read_text()
    assert "[two-0] exit 0" in log.read_text()


def test_a_file_holding_no_report_is_refused(tmp_path: Path) -> None:
    results = tmp_path / "results.txt"
    results.write_text("nothing here\n")

    with pytest.raises(SystemExit, match="no lane's report"):
        main(["--results", str(results), "--log", str(tmp_path / "log")], accept=lambda _: (0, ""))
