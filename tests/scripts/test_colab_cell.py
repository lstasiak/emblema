"""A cell written from a plan, run to its end with the platform's commands stubbed."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.colab_cell import cell_of, faults_of, main, read_plan

COMMIT = "a" * 40
ORDER_A, ORDER_B = "1" * 64, "2" * 64
WEIGHTS = "3" * 64
LABELS = "4" * 64
BLOCK = "5" * 64

STUBS = {
    # A step named by HANG never ends, as a stalled server leaves it.
    "git": 'case " $* " in *" clone "*)\n'
    '  [ "${HANG:-}" = clone ] && exec sleep 100000\n'
    "  mkdir -p emblema/.git; exit 0;;\nesac\necho stub; exit 0\n",
    "pip": '[ "${HANG:-}" = install ] && exec sleep 100000\nexit 0\n',
    "python": 'case "$*" in\n'
    '  *labels.py*) [ "${HANG:-}" = labels ] && exec sleep 100000; exit 0;;\n'
    '  *prefetch.py*) [ "${HANG:-}" = blocks ] && exec sleep 100000; exit 0;;\n'
    "esac\n"
    'echo "$EMBLEMA_WORKER__BACKBONE $*" >> "$CALLS"\n'
    'case "$*" in *"${FAIL:-@none@}"*) exit 3;; esac\n'
    'echo "1 cell answered: durable/sha256/res sha256:res" >&2\n'
    "echo durable/sha256/res sha256:res\n",
    "nvidia-smi": 'case "$*" in *-l*) exec sleep 100000;; esac\necho "card, 1 MiB"\n',
    "nvidia-cuda-mps-control": "exit 0\n",
    "free": 'echo "total used"\necho "Mem: 1 1"\n',
    "nproc": "echo 2\n",
    # macOS has no timeout(1); the alarm's signal ends the command with a non-zero status.
    "timeout": 'seconds=$1; shift\nexec perl -e \'alarm shift; exec @ARGV\' "$seconds" "$@"\n',
}


def write_plan(tmp_path: Path, **lines: str) -> Path:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "orders.txt").write_text(
        f"# from {COMMIT}\nfirst id {ORDER_A}\nsecond id {ORDER_B}\n"
    )
    (plan_dir / "campaigns.txt").write_text(f"first id {WEIGHTS}\nsecond id {WEIGHTS}\n")
    (plan_dir / "worker-env.sh").write_text(
        "# settings\nB=x\nexport EMBLEMA_WORKER__BACKBONE=old\n"
        "export EMBLEMA_WORKER__WORKSPACE=old\nexport EMBLEMA_WORKER__LORA__RANK=8\n"
    )
    (plan_dir / "labels.txt").write_text(f"transient/sha256/{LABELS} sha256:{LABELS}\n")
    defaults = {
        "name": '"emblema-test"',
        "orders": '"orders.txt"',
        "weights": '"campaigns.txt"',
        "worker_env": '"worker-env.sh"',
        "blocks": f'["{BLOCK}"]',
        "labels": '"labels.txt"',
        "workers": "2",
        "threads": "2",
        "extras": '"ml"',
        "pieces": '[["first", 2], ["second", 1]]',
    }
    text = "".join(f"{key} = {value}\n" for key, value in {**defaults, **lines}.items())
    (plan_dir / "plan.toml").write_text(text)
    return plan_dir / "plan.toml"


def run_cell(
    tmp_path: Path, plan: Path, **env: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the plan's cell under the stubs: what it did, and the directory of its results."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        stub = bin_dir / name
        stub.write_text(f"#!/bin/bash\n{body}")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "content"
    (root / "drive" / "MyDrive").mkdir(parents=True, exist_ok=True)
    cell = tmp_path / "cell.sh"
    cell.write_text("".join(cell_of(read_plan(plan)).splitlines(keepends=True)[1:]))
    done = subprocess.run(
        ["/bin/bash", str(cell)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CELL_ROOT": str(root),
            "CALLS": str(tmp_path / "calls.txt"),
            "NET_TIMEOUT": "1",
            "NET_BLOCKS_TIMEOUT": "1",
            **env,
        },
    )
    return done, root / "drive" / "MyDrive" / "emblema-test"


def calls_of(tmp_path: Path) -> list[str]:
    calls = tmp_path / "calls.txt"
    return calls.read_text().splitlines() if calls.exists() else []


def what_the_cell_did(tmp_path: Path, results: Path) -> str:
    """Each call and each lane's report, since a stray retry shows nowhere else after the cell."""
    logs = sorted(results.glob("*.log*")) if results.exists() else []
    reports = results / "results.txt"
    return "\n".join(
        [
            *calls_of(tmp_path),
            reports.read_text() if reports.exists() else "no results",
            *(f"{log.name}: {log.read_text()}" for log in logs),
        ]
    )


def test_every_piece_runs_under_its_backbone_and_settings_the_cell_sets_itself_are_left_out(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path)

    done, results = run_cell(tmp_path, plan)

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 3, what_the_cell_did(tmp_path, results)
    assert all(call.startswith(f"durable/sha256/{WEIGHTS}@sha256:{WEIGHTS}") for call in calls)
    assert sorted(call.split("--lane ")[1] for call in calls) == ["0/1", "0/2", "1/2"]
    assert sum(f"durable/sha256/{ORDER_A}" in call for call in calls) == 2
    assert (results / "results.txt").read_text().count("exit 0") == 3
    cell = cell_of(read_plan(plan))
    assert "EMBLEMA_WORKER__LORA__RANK=8" in cell
    assert "EMBLEMA_WORKER__BACKBONE=old" not in cell
    assert "EMBLEMA_WORKER__WORKSPACE=old" not in cell


def test_run_again_a_cell_repeats_only_what_did_not_end(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    run_cell(tmp_path, plan, FAIL="0/1")
    failed = calls_of(tmp_path)
    assert len(failed) == 4  # the failing lane is tried twice before the cell moves on

    done, results = run_cell(tmp_path, plan)

    assert done.returncode == 0
    again = calls_of(tmp_path)[len(failed) :]
    assert len(again) == 1, what_the_cell_did(tmp_path, results)
    assert "0/1" in again[0]
    # The failed lane logged no answer of its own to resume from, so it starts afresh.
    assert "second-0] exit 0" in (results / "results.txt").read_text()


def test_a_finished_cell_runs_nothing_again(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    run_cell(tmp_path, plan)
    first = len(calls_of(tmp_path))

    run_cell(tmp_path, plan)

    assert len(calls_of(tmp_path)) == first


@pytest.mark.parametrize("step", ["clone", "install", "labels", "blocks"])
def test_a_network_step_that_hangs_ends_the_cell_cleanly_with_no_order_started(
    tmp_path: Path, step: str
) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), HANG=step)

    assert done.returncode == 0, done.stderr
    assert calls_of(tmp_path) == []
    assert "did not finish" in done.stdout


def test_a_cell_that_finds_no_drive_stops_before_anything(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    run_cell(tmp_path, plan, HANG="clone")
    done = subprocess.run(
        ["/bin/bash", "-c", "".join(cell_of(read_plan(plan)).splitlines(keepends=True)[1:])],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"], "CELL_ROOT": str(tmp_path / "nowhere")},
    )

    assert done.returncode == 0
    assert "mount Drive first" in done.stdout


def test_the_generated_cell_breaks_none_of_its_own_rules(tmp_path: Path) -> None:
    assert faults_of(cell_of(read_plan(write_plan(tmp_path)))) == []


@pytest.mark.parametrize(
    ("line", "fault"),
    [
        ("git clone https://example.org", "without a limit"),
        ("pip install -e .", "without a limit"),
        ("python /tmp/prefetch.py x", "without a limit"),
        ("x" * 91, "91 characters"),
    ],
)
def test_a_line_that_breaks_a_rule_is_named(line: str, fault: str) -> None:
    assert any(fault in each for each in faults_of(line))


def test_a_long_setting_is_cut_at_its_commas_into_lines_that_join_back(tmp_path: Path) -> None:
    value = ",".join(str(1000 + index) for index in range(40))
    plan = write_plan(tmp_path)
    settings = plan.parent / "worker-env.sh"
    settings.write_text(f"export EMBLEMA_WORKER__PROBE__PENALTIES={value}\n")

    cell = cell_of(read_plan(plan))

    assert faults_of(cell) == []
    held = [line for line in cell.splitlines() if line.startswith(("P0=", "P1=", "P2="))]
    shown = subprocess.run(
        ["/bin/bash", "-c", "; ".join(held) + '; echo "$P0,$P1${P2:+,$P2}"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert shown.stdout.strip() == value
    assert 'export EMBLEMA_WORKER__PROBE__PENALTIES="$P0,$P1' in cell


@pytest.mark.parametrize(
    "value",
    [
        '"' + ",".join(["1000"] * 40) + '"',
        ",".join(["1000"] * 20) + ",a b," + ",".join(["1"] * 20),
        "x" * 200,
    ],
)
def test_a_long_setting_that_cannot_be_cut_safely_is_refused(tmp_path: Path, value: str) -> None:
    plan = write_plan(tmp_path)
    (plan.parent / "worker-env.sh").write_text(f"export EMBLEMA_WORKER__X={value}\n")

    with pytest.raises(SystemExit, match="cannot be cut"):
        cell_of(read_plan(plan))


def test_a_commented_network_step_or_one_under_a_limit_is_no_fault() -> None:
    assert faults_of("# git clone x\ntimeout 5 git clone x\n") == []


def test_a_cell_is_refused_from_any_commit_but_the_orders(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    out = tmp_path / "cell.sh"

    with pytest.raises(SystemExit, match="the orders at"):
        main(["--plan", str(plan), "--out", str(out)], head="b" * 40)

    assert not out.exists()
    main(["--plan", str(plan), "--out", str(out)], head=COMMIT)
    assert out.read_text().startswith("%%bash\n")


def test_a_piece_without_an_order_is_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, pieces='[["third", 1]]')

    with pytest.raises(SystemExit, match="no order or no weights for third"):
        read_plan(plan)


def test_orders_that_name_no_commit_are_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    (plan.parent / "orders.txt").write_text(f"first id {ORDER_A}\n")

    with pytest.raises(SystemExit, match="names no commit"):
        read_plan(plan)


def test_labels_whose_key_does_not_match_their_digest_are_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    (plan.parent / "labels.txt").write_text(f"transient/sha256/{BLOCK} sha256:{LABELS}\n")

    with pytest.raises(SystemExit, match="does not match"):
        read_plan(plan)


def test_a_plan_without_labels_or_blocks_leaves_those_steps_out(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, blocks="[]")
    text = plan.read_text().replace('labels = "labels.txt"\n', "")
    plan.write_text(text)

    cell = cell_of(read_plan(plan))

    assert "labels.py" not in cell
    assert "prefetch.py" not in cell
