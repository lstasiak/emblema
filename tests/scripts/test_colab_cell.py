"""A cell written from a plan, run to its end with the platform's commands stubbed."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.cell_parts import COLAB, KAGGLE, MAC, Platform, faults_of, write_cells
from scripts.colab_cell import cell_of, main, read_plan

COMMIT = "a" * 40
ORDER_A, ORDER_B = "1" * 64, "2" * 64
WEIGHTS = "3" * 64
LABELS = "4" * 64
BLOCK = "5" * 64

STUBS = {
    # A step named by HANG never ends, as a stalled server leaves it.
    "git": 'case " $* " in *" clone "*)\n'
    '  [ "${HANG:-}" = clone ] && exec sleep 100000\n'
    "  mkdir -p emblema/.git; exit 0;;\n"
    '  *" rev-parse HEAD "*) echo "${AT_COMMIT:-"'
    "$COMMIT_HERE"
    '"}"; exit 0;;\n'
    '  *" status "*) echo "${DIRTY:-}"; exit 0;;\n'
    "esac\necho stub; exit 0\n",
    # The Mac's python goes through uv; the arguments before the command are uv's own.
    "uv": 'shift 4\nexec "$(dirname "$0")/python" "$@"\n',
    "caffeinate": 'echo caffeinated >> "$CALLS.caffeinate"\nshift\nexec "$@"\n',
    "pip": '[ "${HANG:-}" = install ] && exec sleep 100000\nexit 0\n',
    "python": 'case "$*" in\n'
    '  *fetch_published*labels*) [ "${HANG:-}" = labels ] && exec sleep 100000; exit 0;;\n'
    '  *fetch_published*blocks*) echo "the store said no" >&2\n'
    '    [ "${HANG:-}" = blocks ] && exec sleep 100000; exit 0;;\n'
    '  *fetch_corpora*) echo "the mirror said no" >&2\n'
    '    [ "${HANG:-}" = corpora ] && exec sleep 100000; exit 0;;\n'
    "esac\n"
    'echo "$EMBLEMA_WORKER__BACKBONE card=${CUDA_VISIBLE_DEVICES:-shared} $*" >> "$CALLS"\n'
    '[ -n "${HOLD:-}" ] && sleep "$HOLD"\n'
    'case "$*" in *"${FAIL:-@none@}"*) exit 3;; esac\n'
    'echo "1 cell answered: durable/sha256/res sha256:res" >&2\n'
    "echo durable/sha256/res sha256:res\n",
    "nvidia-smi": 'case "$*" in *-l*) exec sleep 100000;; esac\necho "card, 1 MiB"\n',
    "nvidia-cuda-mps-control": 'echo "$*" >> "$CALLS.mps"\n',
    # As procps on Linux: a name is its first 15 characters, a command line is whole.
    "pgrep": "running=${MPS_RUNNING:-}\n"
    '[ "$1" = -f ] && { [ -n "$running" ] && echo "$running" | grep -qE "$2"; exit; }\n'
    "name=${running%% *}; name=${name##*/}\n"
    '[ "$1" = -x ] && [ -n "$running" ] && [ "$2" = "${name:0:15}" ]\n',
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
    tmp_path: Path, plan: Path, platform: Platform = COLAB, **env: str
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
    cell.write_text("".join(cell_of(read_plan(plan), platform).splitlines(keepends=True)[1:]))
    work = tmp_path / "working"
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
    results = root / "drive" / "MyDrive" / "emblema-test"
    if platform is MAC:
        return done, root / "data" / "report" / "emblema-test"
    return done, work / "emblema-test" if platform is KAGGLE else results


def calls_of(tmp_path: Path) -> list[str]:
    calls = tmp_path / "calls.txt"
    return calls.read_text().splitlines() if calls.exists() else []


def what_the_cell_did(tmp_path: Path, results: Path) -> str:
    """Each call and each lane's report, since a stray run shows nowhere else after the cell."""
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


def test_two_workers_never_run_one_piece_where_mkdir_lets_both_through(tmp_path: Path) -> None:
    # uutils mkdir answers success to a directory that came to exist while it worked; -p does so
    # always, so this holds on every machine. The hold keeps each piece running while the other
    # worker reaches it.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real = shutil.which("mkdir")
    assert real is not None
    stub = bin_dir / "mkdir"
    stub.write_text(f'#!/bin/bash\nexec {real} -p "$@"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    done, results = run_cell(tmp_path, write_plan(tmp_path), HOLD="0.3")

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 3, what_the_cell_did(tmp_path, results)
    assert sorted(call.split("--lane ")[1] for call in calls) == ["0/1", "0/2", "1/2"]


def test_run_again_a_cell_repeats_only_what_did_not_end(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    first, results = run_cell(tmp_path, plan, FAIL="0/1")
    failed = calls_of(tmp_path)
    # The failing lane is tried twice before the cell moves on; a cell that stopped before its
    # workers says why on its output.
    assert len(failed) == 4, f"{first.stdout}\n{what_the_cell_did(tmp_path, results)}"

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
        ("python scripts/fetch_published.py blocks x", "without a limit"),
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


def test_a_kaggle_cell_spreads_its_workers_over_the_cards_and_needs_no_drive(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path, cards="2", workers="4", pieces='[["first", 4]]')

    done, results = run_cell(tmp_path, plan, KAGGLE, HOLD="0.5")

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 4, what_the_cell_did(tmp_path, results)
    assert all(call.startswith(f"durable/sha256/{WEIGHTS}@sha256:{WEIGHTS}") for call in calls)
    assert sorted(call.split()[1] for call in calls) == [
        "card=0",
        "card=0",
        "card=1",
        "card=1",
    ]
    assert (results / "results.txt").read_text().count("exit 0") == 4


@pytest.mark.parametrize(
    "running", ["nvidia-cuda-mps-control -d", "/usr/bin/nvidia-cuda-mps-control -d"]
)
def test_a_cell_run_again_finds_the_mps_daemon_it_left(tmp_path: Path, running: str) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MPS_RUNNING=running)

    assert "MPS: already running" in done.stdout
    assert "-d" not in Path(f"{tmp_path / 'calls.txt'}.mps").read_text().split()


@pytest.mark.parametrize("running", ["bash cell.sh", "bash -c nvidia-cuda-mps-control -d"])
def test_a_cell_starts_the_mps_daemon_when_none_runs(tmp_path: Path, running: str) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MPS_RUNNING=running)

    assert "MPS: started" in done.stdout
    assert Path(f"{tmp_path / 'calls.txt'}.mps").read_text().splitlines()[0] == "-d"


@pytest.mark.parametrize("platform", [COLAB, KAGGLE, MAC])
def test_a_plan_with_a_long_name_makes_a_cell_within_the_line_limit(
    tmp_path: Path, platform: Platform
) -> None:
    plan = write_plan(tmp_path, name=f'"{"l46/trial/" + "x" * 30}"', cards="2")
    assert faults_of(cell_of(read_plan(plan), platform)) == []


@pytest.mark.parametrize("platform", [COLAB, KAGGLE])
def test_the_cells_beside_a_cell_keep_to_its_line_limit(tmp_path: Path, platform: Platform) -> None:
    written = write_cells(tmp_path / "cell.sh", "true\n", platform)

    assert len(written) > 1
    assert all(faults_of(path.read_text()) == [] for path in written)


def test_a_kaggle_cell_starts_no_mps_daemon(tmp_path: Path) -> None:
    cell = cell_of(read_plan(write_plan(tmp_path, cards="2")), KAGGLE)
    assert "nvidia-cuda-mps-control -d" not in cell
    assert "/kaggle/working" in cell


def test_a_kaggle_cell_without_cards_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="spreads its workers over cards"):
        cell_of(read_plan(write_plan(tmp_path)), KAGGLE)


def test_a_colab_cell_shares_one_card(tmp_path: Path) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path))

    assert done.returncode == 0, done.stderr
    assert all(call.split()[1] == "card=shared" for call in calls_of(tmp_path))


def test_past_the_cutoff_no_piece_is_claimed(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, cards="2", pieces='[["first", 2]]')

    done, results = run_cell(tmp_path, plan, KAGGLE, SESSION_SECONDS="-1")

    assert done.returncode == 0, done.stderr
    assert calls_of(tmp_path) == []
    assert (results / "results.txt").read_text().count("not claimed: past the cutoff") == 2


def test_a_kaggle_cell_stops_claiming_at_ten_and_a_half_hours(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, cards="2")
    assert "CUTOFF=$(( START + ${SESSION_SECONDS:-37800} ))" in cell_of(read_plan(plan), KAGGLE)


def test_a_corpora_fetch_that_hangs_ends_the_cell_cleanly(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, fetch='[["skab", "opssat_ad"]]')

    done, results = run_cell(tmp_path, plan, HANG="corpora")

    assert done.returncode == 0
    assert "corpora not fetched" in done.stdout
    assert calls_of(tmp_path) == []
    assert "the mirror said no" in (results / "fetch.log").read_text()


def test_the_blocks_stderr_is_kept_for_the_person_reading_a_failure(tmp_path: Path) -> None:
    done, results = run_cell(tmp_path, write_plan(tmp_path), HANG="blocks")

    assert done.returncode == 0
    assert "blocks not fetched" in done.stdout
    assert "the store said no" in (results / "block.log").read_text()


@pytest.mark.parametrize("platform", ["colab", "kaggle"])
def test_the_cells_around_the_cell_are_written_beside_it(tmp_path: Path, platform: str) -> None:
    plan = write_plan(tmp_path, cards="2")
    out = tmp_path / "cell.sh"
    main(["--plan", str(plan), "--out", str(out), "--platform", platform], head=COMMIT)
    secrets = (tmp_path / "cell-secrets.py").read_text()
    assert "EMBLEMA_ARTIFACT_STORE__" in secrets
    assert ("userdata" in secrets) == (platform == "colab")
    assert (tmp_path / "cell-release.py").exists() == (platform == "colab")


def test_a_fetch_with_many_arguments_stays_within_the_line_limit(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, fetch='[["--from-store", "physionet2012", "physionet2019"]]')
    assert faults_of(cell_of(read_plan(plan))) == []


def test_the_last_answer_is_taken_from_the_newest_log(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, pieces='[["first", 1]]')
    _, results = run_cell(tmp_path, plan)
    results.mkdir(parents=True, exist_ok=True)
    (results / "results.txt").unlink()
    (results / "first-0.log.1000").write_text("1 cell answered: durable/sha256/aaa sha256:aaa\n")
    (results / "first-0.log.2000").write_text("1 cell answered: durable/sha256/bbb sha256:bbb\n")
    (results / "first-0.log").write_text("1 cell answered: durable/sha256/ccc sha256:ccc\n")
    (tmp_path / "calls.txt").unlink()

    run_cell(tmp_path, plan)

    assert "--resume durable/sha256/ccc sha256:ccc" in calls_of(tmp_path)[0]


def test_a_piece_still_running_past_the_grace_is_stopped_and_not_tried_again(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path, cards="1", workers="1", pieces='[["first", 1]]')

    done, results = run_cell(
        tmp_path, plan, KAGGLE, SESSION_SECONDS="12", GRACE_SECONDS="1", HOLD="60"
    )

    assert done.returncode == 0, done.stderr
    assert len(calls_of(tmp_path)) == 1
    assert "exit 0" not in (results / "results.txt").read_text()


def test_a_step_with_its_limit_after_it_is_a_fault() -> None:
    assert "without a limit" in faults_of("true; curl x; timeout 5 true")[0]


@pytest.mark.parametrize(
    "line", ["pip3 install x", "git fetch origin", "python scripts/fetch_published.py labels x"]
)
def test_the_other_forms_of_a_network_step_need_a_limit_too(line: str) -> None:
    assert "without a limit" in faults_of(line)[0]


def test_on_the_mac_a_script_runs_the_orders_on_mps_through_uv_and_keeps_awake(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path, workers="1", pieces='[["first", 2]]')

    done, results = run_cell(tmp_path, plan, MAC)

    assert done.returncode == 0, done.stderr
    calls = calls_of(tmp_path)
    assert len(calls) == 2, what_the_cell_did(tmp_path, results)
    assert (tmp_path / "calls.txt.caffeinate").read_text() == "caffeinated\n"
    assert (results / "results.txt").read_text().count("exit 0") == 2
    cell = cell_of(read_plan(plan), MAC)
    assert cell.startswith("#!/bin/bash")
    assert "EMBLEMA_WORKER__DEVICE=mps" in cell
    assert "nvidia-smi" not in cell
    assert "git clone" not in cell


def test_on_the_mac_a_script_refuses_another_commit(tmp_path: Path) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MAC, AT_COMMIT="b" * 40)

    assert "the orders at" in done.stdout
    assert calls_of(tmp_path) == []


def test_on_the_mac_a_script_refuses_a_dirty_tree(tmp_path: Path) -> None:
    done, _ = run_cell(tmp_path, write_plan(tmp_path), MAC, DIRTY=" M scripts/x.py")

    assert "dirty" in done.stdout
    assert calls_of(tmp_path) == []


def test_on_the_mac_untracked_files_count_as_dirty_as_the_order_counts_them(
    tmp_path: Path,
) -> None:
    assert 'git status --porcelain)"' in cell_of(read_plan(write_plan(tmp_path)), MAC)


def test_on_the_mac_the_raw_corpora_are_not_fetched(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, fetch='[["skab"]]')
    cell = cell_of(read_plan(plan), MAC)
    assert "fetch_corpora" not in cell
    assert "fetch_published.py labels" not in cell
    assert "fetch_published.py blocks" in cell


def test_on_the_mac_nothing_is_written_beside_the_script(tmp_path: Path) -> None:
    out = tmp_path / "run.sh"
    plan = write_plan(tmp_path)
    main(["--plan", str(plan), "--out", str(out), "--platform", "mac"], head=COMMIT)
    assert out.stat().st_mode & stat.S_IXUSR
    assert not (tmp_path / "run-secrets.py").exists()
    assert not (tmp_path / "run-release.py").exists()
