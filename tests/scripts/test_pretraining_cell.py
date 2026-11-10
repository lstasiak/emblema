"""A pretraining cell written from a plan, run to its end with the platform's commands stubbed."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from emblema.entrypoints.cli.pretrain.pretrain_cli import PretrainCli
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from scripts.cell_parts import COLAB, KAGGLE, MAC, Platform, faults_of
from scripts.pretraining_cell import cell_of, main, read_plan

RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "src/emblema/pretraining/adapters/training/torch_training_runtime.py"
)
COMMIT = "a" * 40
ORDER_A, ORDER_B = "1" * 64, "2" * 64
BLOCK = "5" * 64
CHECKPOINT = f"transient/sha256/{'c' * 64} sha256:{'c' * 64}"

STUBS = {
    "git": 'case " $* " in *" clone "*)\n'
    '  [ "${HANG:-}" = clone ] && exec sleep 100000\n'
    "  mkdir -p emblema/.git; exit 0;;\n"
    '  *" rev-parse HEAD "*) echo "${AT_COMMIT:-"'
    "$COMMIT_HERE"
    '"}"; exit 0;;\n'
    '  *" --show-toplevel "*) [ -n "${NO_REPO:-}" ] && exit 128; pwd; exit 0;;\n'
    '  *" status "*) echo "${DIRTY:-}"; exit 0;;\n'
    "esac\necho stub; exit 0\n",
    # The Mac's python goes through uv; the arguments before the command are uv's own.
    "uv": 'shift 4\nexec "$(dirname "$0")/python" "$@"\n',
    "caffeinate": 'echo caffeinated >> "$CALLS.caffeinate"\nshift\nexec "$@"\n',
    "pip": '[ "${HANG:-}" = install ] && exec sleep 100000\nexit 0\n',
    "python": 'case "$*" in\n'
    '  *fetch_published*blocks*) [ "${HANG:-}" = blocks ] && exec sleep 100000; exit 0;;\n'
    "esac\n"
    'echo "$* card=${CUDA_VISIBLE_DEVICES:-none}" >> "$CALLS"\n'
    f'echo "checkpoint {CHECKPOINT}" >&2\n'
    '[ -n "${HOLD:-}" ] && sleep "$HOLD"\n'
    'case "$*" in *"${FAIL:-@none@}"*) exit 3;; esac\n'
    "echo the-result\n",
    "nvidia-smi": 'case "$*" in *-l*) exec sleep 100000;; esac\necho "card, 1 MiB"\n',
    "nproc": "echo 2\n",
    "timeout": 'seconds=$1; shift\nexec perl -e \'alarm shift; exec @ARGV\' "$seconds" "$@"\n',
}


def write_plan(tmp_path: Path, *extra: str, **lines: str) -> Path:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "orders.txt").write_text(
        f"# from {COMMIT}\nfirst id durable/sha256/x sha256:{ORDER_A}\nsecond id {ORDER_B}\n"
    )
    defaults = {
        "name": '"emblema-test"',
        "orders": '"orders.txt"',
        "pieces": '["first", "second"]',
        "cards": "1",
    }
    text = "".join(f"{key} = {value}\n" for key, value in {**defaults, **lines}.items())
    (plan_dir / "plan.toml").write_text(text + "".join(f"{each}\n" for each in extra))
    return plan_dir / "plan.toml"


def run_cell(
    tmp_path: Path, plan: Path, platform: Platform = COLAB, **env: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        stub = bin_dir / name
        stub.write_text(f"#!/bin/bash\n{body}")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "content"
    (root / "drive" / "MyDrive").mkdir(parents=True, exist_ok=True)
    work = tmp_path / "working"
    cell = tmp_path / "cell.sh"
    cell.write_text("".join(cell_of(read_plan(plan), platform).splitlines(keepends=True)[1:]))
    done = subprocess.run(
        ["/bin/bash", str(cell)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CELL_ROOT": str(root),
            "CELL_WORK": str(work),
            "COMMIT_HERE": COMMIT,
            "CALLS": str(tmp_path / "calls.txt"),
            "NET_TIMEOUT": "1",
            "NET_BLOCKS_TIMEOUT": "1",
            **env,
        },
    )
    if platform is MAC:
        return done, root / "data" / "report" / "emblema-test"
    results = root / "drive" / "MyDrive" / "emblema-test"
    return done, work / "emblema-test" if platform is KAGGLE else results


def calls_of(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_each_order_runs_once_under_the_commit_and_the_workspace(tmp_path: Path) -> None:
    done, results = run_cell(tmp_path, write_plan(tmp_path))

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 2
    assert f"durable/sha256/{ORDER_A} sha256:{ORDER_A}" in calls[0]
    assert f"durable/sha256/{ORDER_B} sha256:{ORDER_B}" in calls[1]
    assert all(f"--commit {COMMIT}" in call and "--workspace" in call for call in calls)
    assert all("--device cuda" in call and "--progress-every 100" in call for call in calls)
    assert (results / "results.txt").read_text().count("exit 0") == 2


def test_the_first_run_starts_from_the_beginning(tmp_path: Path) -> None:
    run_cell(tmp_path, write_plan(tmp_path))

    assert not any("--resume-from" in call for call in calls_of(tmp_path))


def test_an_attempt_that_fails_resumes_from_its_own_checkpoint(tmp_path: Path) -> None:
    done, results = run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]'), FAIL=ORDER_A)

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 2
    assert "--resume-from" not in calls[0]
    assert f"--resume-from {CHECKPOINT}" in calls[1]
    assert (results / "results.txt").read_text().count("exit 3") == 2
    assert len(list(results.glob("first.log*"))) == 2


def test_run_again_a_cell_resumes_what_failed_and_repeats_nothing_else(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    run_cell(tmp_path, plan, FAIL=ORDER_B)
    before = len(calls_of(tmp_path))
    done, _ = run_cell(tmp_path, plan)

    assert done.returncode == 0, done.stderr
    again = calls_of(tmp_path)[before:]
    assert len(again) == 1
    assert ORDER_B in again[0]
    assert f"--resume-from {CHECKPOINT}" in again[0]


def test_a_checkpoint_from_another_session_is_taken_from_the_plan(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, "[resume]", f'first = "{CHECKPOINT}"', pieces='["first"]')

    run_cell(tmp_path, plan)

    assert f"--resume-from {CHECKPOINT}" in calls_of(tmp_path)[0]
    assert faults_of(cell_of(read_plan(plan), COLAB)) == []


def test_a_cell_prints_where_an_unfinished_order_can_continue(tmp_path: Path) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]'), FAIL=ORDER_A)

    assert f"first {CHECKPOINT}" in done.stdout


def test_each_worker_keeps_to_its_own_card(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, cards="2")

    done, _ = run_cell(tmp_path, plan, KAGGLE, HOLD="0.5")

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert sorted(call.rsplit(" ", 1)[1] for call in calls) == ["card=0", "card=1"]


@pytest.mark.parametrize("step", ["clone", "install", "blocks"])
def test_a_network_step_that_hangs_ends_the_cell_with_no_order_started(
    tmp_path: Path, step: str
) -> None:
    plan = write_plan(tmp_path, blocks=f'["{BLOCK}"]')

    done, _ = run_cell(tmp_path, plan, HANG=step)

    assert done.returncode == 0
    assert calls_of(tmp_path) == []


def test_a_colab_cell_that_finds_no_drive_stops_before_anything(tmp_path: Path) -> None:
    empty = tmp_path / "no-drive"
    empty.mkdir()

    done, _ = run_cell(tmp_path, write_plan(tmp_path), CELL_ROOT=str(empty))

    assert done.returncode == 0
    assert "mount Drive first" in done.stdout
    assert calls_of(tmp_path) == []


@pytest.mark.parametrize("platform", [COLAB, KAGGLE, MAC])
def test_the_generated_cell_breaks_none_of_its_own_rules(
    tmp_path: Path, platform: Platform
) -> None:
    plan = write_plan(tmp_path, blocks=f'["{BLOCK}"]')
    assert faults_of(cell_of(read_plan(plan), platform)) == []


def test_a_cell_is_refused_from_any_commit_but_the_orders(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    with pytest.raises(SystemExit, match="the tree is at"):
        main(["--plan", str(plan), "--out", str(tmp_path / "c.sh")], head_commit="b" * 40)


def test_orders_that_name_no_commit_are_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    (plan.parent / "orders.txt").write_text("first id x\n")
    with pytest.raises(SystemExit, match="names no commit"):
        read_plan(plan)


def test_a_piece_without_an_order_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no order for third"):
        read_plan(write_plan(tmp_path, pieces='["first", "third"]'))


def test_a_malformed_checkpoint_is_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, "[resume]", 'first = "durable/sha256/x sha256:y"')
    with pytest.raises(SystemExit, match="not a checkpoint"):
        read_plan(plan)


def test_a_checkpoint_for_no_piece_is_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, "[resume]", f'third = "{CHECKPOINT}"')
    with pytest.raises(SystemExit, match="no piece"):
        read_plan(plan)


def test_a_plan_with_no_card_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs a card"):
        read_plan(write_plan(tmp_path, cards="0"))


@pytest.mark.parametrize("platform", ["colab", "kaggle"])
def test_the_cells_around_the_cell_are_written_beside_it(tmp_path: Path, platform: str) -> None:
    out = tmp_path / "cell.sh"
    main(
        ["--plan", str(write_plan(tmp_path)), "--out", str(out), "--platform", platform],
        head_commit=COMMIT,
    )
    assert (tmp_path / "cell-secrets.py").exists()
    assert (tmp_path / "cell-release.py").exists() == (platform == "colab")


def test_the_newest_checkpoint_is_taken_whatever_the_logs_are_called(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, pieces='["first"]')
    _, results = run_cell(tmp_path, plan, FAIL=ORDER_A)
    (results / "results.txt").unlink()
    newest = f"transient/sha256/{'d' * 64} sha256:{'d' * 64}"
    older = f"transient/sha256/{'b' * 64} sha256:{'b' * 64}"
    for log in results.glob("first.log*"):
        log.unlink()
    (results / "first.log.1000").write_text(f"checkpoint {older}\n")
    (results / "first.log.2000").write_text(f"checkpoint {older}\n")
    (results / "first.log").write_text(f"checkpoint {newest}\n")
    (tmp_path / "calls.txt").unlink()

    run_cell(tmp_path, plan)

    assert f"--resume-from {newest}" in calls_of(tmp_path)[0]


def test_a_kaggle_order_still_running_past_the_grace_is_stopped_and_names_its_checkpoint(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path, pieces='["first"]')

    done, results = run_cell(
        tmp_path, plan, KAGGLE, SESSION_SECONDS="12", GRACE_SECONDS="1", HOLD="60"
    )

    assert done.returncode == 0, done.stderr
    assert len(calls_of(tmp_path)) == 1
    assert f"first {CHECKPOINT}" in done.stdout
    assert "exit 0" not in (results / "results.txt").read_text()


def test_the_cell_asks_the_pretraining_command_line_for_what_it_accepts(tmp_path: Path) -> None:
    run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]'))
    words = calls_of(tmp_path)[0].split()
    arguments = words[words.index("run") :]
    arguments = [word for word in arguments if not word.startswith("card=")]

    invocation = PretrainCli().parse(arguments)

    assert isinstance(invocation.command, FulfilPretrainingOrderCommand)
    assert (invocation.device, invocation.progress_every) == ("cuda", 100)


def test_the_checkpoint_pattern_is_what_the_training_runtime_logs() -> None:
    source = RUNTIME.read_text()
    assert '"checkpoint %s %s at step' in source


def test_several_orders_share_a_card_at_once_when_the_plan_says_so(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, per_card="2", threads="4")

    done, _ = run_cell(tmp_path, plan, HOLD="1")

    assert done.returncode == 0, done.stderr
    assert len(calls_of(tmp_path)) == 2
    cell = cell_of(read_plan(plan))
    assert "nvidia-cuda-mps-control -d" in cell
    assert "export OMP_NUM_THREADS=4" in cell


def test_a_card_held_by_one_order_starts_no_mps(tmp_path: Path) -> None:
    assert "nvidia-cuda-mps-control -d" not in cell_of(read_plan(write_plan(tmp_path)))


@pytest.mark.parametrize("platform", [KAGGLE, MAC])
def test_a_card_is_not_shared_where_the_platform_does_not_do_it(
    tmp_path: Path, platform: Platform
) -> None:
    with pytest.raises(SystemExit, match="does not share a card"):
        cell_of(read_plan(write_plan(tmp_path, per_card="2", threads="2")), platform)


def test_a_plan_asks_for_batch_collating_processes(tmp_path: Path) -> None:
    run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]', num_workers="4"))

    assert "--num-workers 4" in calls_of(tmp_path)[0]


def test_no_collating_processes_are_asked_for_by_default(tmp_path: Path) -> None:
    run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]'))

    assert "--num-workers" not in calls_of(tmp_path)[0]


def test_on_the_mac_a_pretraining_runs_on_mps_through_uv(tmp_path: Path) -> None:
    done, results = run_cell(tmp_path, write_plan(tmp_path, pieces='["first"]'), MAC)

    assert done.returncode == 0, done.stderr
    assert "--device mps" in calls_of(tmp_path)[0]
    assert (results / "results.txt").read_text().count("exit 0") == 1
    assert (tmp_path / "calls.txt.caffeinate").exists()


def test_on_the_mac_a_failed_run_resumes_when_the_script_is_run_again(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, pieces='["first"]')
    run_cell(tmp_path, plan, MAC, FAIL=ORDER_A)
    before = len(calls_of(tmp_path))

    run_cell(tmp_path, plan, MAC)

    assert f"--resume-from {CHECKPOINT}" in calls_of(tmp_path)[before]


@pytest.mark.parametrize("env", [{"CELL_ROOT": "", "NO_REPO": "1"}, {"CELL_ROOT": "/no/such"}])
def test_on_the_mac_a_script_found_outside_a_repository_does_nothing(
    tmp_path: Path, env: dict[str, str]
) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MAC, **env)

    assert "no repository" in done.stdout
    assert calls_of(tmp_path) == []


@pytest.mark.parametrize("platform", [COLAB, KAGGLE, MAC])
def test_a_plan_with_a_long_name_makes_a_cell_within_the_line_limit(
    tmp_path: Path, platform: Platform
) -> None:
    plan = write_plan(tmp_path, name=f'"{"l46/trial/" + "x" * 30}"')
    assert faults_of(cell_of(read_plan(plan), platform)) == []


def test_on_the_mac_a_script_refuses_a_dirty_tree(tmp_path: Path) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MAC, DIRTY=" M scripts/x.py")

    assert "dirty" in done.stdout
    assert calls_of(tmp_path) == []


def test_orders_sharing_a_card_must_name_their_threads(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="limit on threads"):
        read_plan(write_plan(tmp_path, per_card="2"))


def test_the_mac_refuses_more_than_one_card(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="one accelerator"):
        cell_of(read_plan(write_plan(tmp_path, cards="2")), MAC)
