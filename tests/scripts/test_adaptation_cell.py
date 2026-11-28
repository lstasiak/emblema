"""The forecaster's fine-tuning cell: its plan, its own rules and the pieces it runs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.adaptation_cell import (
    _WORK,
    ADAPTERS,
    WEIGHTS,
    AdaptationPlan,
    cell_of,
    main,
    pieces_of,
    read_plan,
    wrapped,
)
from scripts.cell_parts import faults_of
from scripts.put_archive import files_of

COMMIT = "a" * 40
INPUTS = "b" * 64


def plan_files(tmp_path: Path, *, per_card: int = 3, commit: str = COMMIT) -> Path:
    (tmp_path / "fits.txt").write_text(
        "# the dose\n"
        "r8 fit --farm penmanshiel --hourly @IN@/hourly/penmanshiel --out @F@/fits/r8\n"
        "r64 fit --farm penmanshiel --rank 64 --out @F@/fits/r64\n"
    )
    (tmp_path / "answers.txt").write_text(
        "a-r8 answer --fits @F@/fits/r8 --cells @IN@/dose/answers-penmanshiel-N2.csv\n"
    )
    plan = tmp_path / "plan.toml"
    plan.write_text(
        f'name = "emblema-adaptation"\ncommit = "{commit}"\ninputs = "{INPUTS}"\n'
        f'per_card = {per_card}\nfits = "fits.txt"\nanswers = "answers.txt"\n'
    )
    return plan


def test_the_plan_reads_its_pieces_and_the_cell_keeps_its_rules(tmp_path: Path) -> None:
    plan = read_plan(plan_files(tmp_path))
    cell = cell_of(plan)

    assert [line.split()[0] for line in plan.fits] == ["r8", "r64"]
    assert faults_of(cell) == []
    assert cell.startswith("%%bash")
    assert "git checkout -q $C" in cell
    assert f"C={COMMIT}" in cell
    assert f"INPUTS={INPUTS}" in cell
    assert "fetch_published.py archive $INPUTS $I" in cell
    assert "chronos-forecasting==2.3.2 peft==0.18.1" in cell
    assert "export HF_HUB_OFFLINE=1" in cell
    assert "nvidia-cuda-mps-control -d" in cell
    assert "--exclude '*/model.safetensors' '*.bin' '*.pt' '*.pth'" in cell
    assert "pip install -q -e . 2>" in cell
    assert "nvidia-cuda-mps-control -d" not in cell_of(read_plan(plan_files(tmp_path, per_card=1)))


def test_a_malformed_plan_or_piece_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="40 hex"):
        read_plan(plan_files(tmp_path, commit="abc"))
    with pytest.raises(SystemExit, match="named once"):
        pieces_of("a fit x\na fit y\n")
    with pytest.raises(SystemExit, match="named once"):
        pieces_of("a/b fit x\n")
    with pytest.raises(SystemExit, match="no quote"):
        pieces_of("a fit --out '$(rm -rf /)'\n")
    plan = plan_files(tmp_path)
    (tmp_path / "answers.txt").write_text("r8 answer --fits @F@/fits/r8\n")
    with pytest.raises(SystemExit, match=r"share the labels \['r8'\]"):
        read_plan(plan)
    plan = plan_files(tmp_path)
    plan.write_text(plan.read_text().replace('"emblema-adaptation"', '"two words"'))
    with pytest.raises(SystemExit, match="no plain directory name"):
        read_plan(plan)
    (tmp_path / "empty.txt").write_text("# nothing\n")
    plan = plan_files(tmp_path)
    plan.write_text(plan.read_text().replace('fits = "fits.txt"', 'fits = "empty.txt"'))
    with pytest.raises(SystemExit, match="fits nothing"):
        read_plan(plan)


def test_the_cell_is_written_only_at_the_plans_commit(tmp_path: Path) -> None:
    plan = plan_files(tmp_path)
    out = tmp_path / "cell.sh"
    with pytest.raises(SystemExit, match="the tree is at"):
        main(["--plan", str(plan), "--out", str(out)], head_commit="c" * 40)
    main(["--plan", str(plan), "--out", str(out)], head_commit=COMMIT)
    assert out.exists()
    assert out.with_name("cell-release.py").exists()


def work(tmp_path: Path, plan: AdaptationPlan, *, failing: str = "") -> list[str]:
    """Run the cell's work on stand-ins; the calls the fine-tuning script saw, in order."""
    work, scripts, bin_ = tmp_path / "w", tmp_path / "scripts", tmp_path / "bin"
    for directory in (work, scripts, bin_):
        directory.mkdir(exist_ok=True)
    # Stand-ins: the fine-tuning script records its call and fails on the label named failing;
    # python runs a script, -I dropped; the archive prints a key.
    (scripts / "fleet_chronos2_adapted.py").write_text(
        "import os, sys\n"
        f"open({str(tmp_path / 'calls.txt')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "failing = os.environ['FAILING']\n"
        "fails = failing and sys.argv[1] == 'fit' and sys.argv[-3].endswith(failing)\n"
        "sys.exit(1 if fails else 0)\n"
    )
    (scripts / "put_archive.py").write_text("print('transient/sha256/k sha256:k')\n")
    (bin_ / "python").write_text(
        f'#!/bin/bash\n[ "$1" = -I ] && shift\nexec {sys.executable} "$@"\n'
    )
    (bin_ / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
    for stand_in in ("python", "timeout"):
        (bin_ / stand_in).chmod(0o755)
    (work / "fits.txt").write_text("\n".join(x for p in plan.fits for x in wrapped(p)) + "\n")
    (work / "answers.txt").write_text("\n".join(plan.answers) + "\n")
    (work / "machine.txt").write_text("card\n")
    body = (
        _WORK.replace("@PER_CARD@", "2")
        .replace("@MONITOR@", "")
        .replace("@STOP@", "")
        .replace("@UTILISATION@", "")
        .replace("@WEIGHTS@", "'*.bin'")
        .replace("@ADAPTERS@", ADAPTERS)
    )
    script = f"set -u\nW={work}\nF={tmp_path / 'f'}\nI=/inputs\nT=$(mktemp -d)\nSTART=0\nNET=5\n"
    environment = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAILING": failing}
    shown = subprocess.run(
        ["bash", "-c", script + body],
        cwd=tmp_path,
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    ).stdout
    assert "transient/sha256/k" in shown
    record = tmp_path / "calls.txt"
    calls = record.read_text().splitlines() if record.exists() else []
    record.unlink(missing_ok=True)
    return calls


PLAN = AdaptationPlan(
    name="n",
    commit=COMMIT,
    inputs=INPUTS,
    per_card=2,
    fits=(
        "f1 fit --hourly @IN@/h --glob * --out @F@/fits/f1 " + " ".join(["--x 1"] * 30),
        "f2 fit --out @F@/fits/f2",
    ),
    answers=(
        "a1 answer --fits @F@/fits/f1 --out @W@/answers/a1.csv",
        "a2 answer --fits @F@/fits/f2",
    ),
)


def test_the_workers_run_every_fit_once_then_the_answers_with_the_paths_filled(
    tmp_path: Path,
) -> None:
    calls = work(tmp_path, PLAN)

    f = tmp_path / "f"
    long = " ".join(["--x 1"] * 30)
    assert sorted(calls[:2]) == [
        f"fit --hourly /inputs/h --glob * --out {f}/fits/f1 {long} --device cuda",
        f"fit --out {f}/fits/f2 --device cuda",
    ]
    assert sorted(calls[2:]) == [
        f"answer --fits {f}/fits/f1 --out {tmp_path}/w/answers/a1.csv --device cuda",
        f"answer --fits {f}/fits/f2 --device cuda",
    ]
    assert work(tmp_path, PLAN) == []


def test_a_failed_fit_holds_its_answers_and_a_run_again_redoes_only_what_is_not_done(
    tmp_path: Path,
) -> None:
    first = work(tmp_path, PLAN, failing="f2")
    again = work(tmp_path, PLAN)
    with (tmp_path / "w" / "results.txt").open("a") as results:
        results.write("[f1] exit 0 in 1 s\n")
    after_refit = work(tmp_path, PLAN)

    assert sorted(c.split()[0] for c in first) == ["answer", "fit", "fit"]
    assert all("fits/f2" not in c for c in first if c.startswith("answer"))
    assert sorted(" ".join(c.split()[:3]) for c in again) == [
        f"answer --fits {tmp_path}/f/fits/f2",
        f"fit --out {tmp_path}/f/fits/f2",
    ]
    assert after_refit == [
        f"answer --fits {tmp_path}/f/fits/f1 --out {tmp_path}/w/answers/a1.csv --device cuda"
    ]


def test_each_checkpoints_adapters_and_records_go_back_and_no_full_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "f" / "fits" / "f1" / "checkpoint-8000"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        ADAPTERS,
        "optimizer.pt",
        "rng_state.pth",
        "model.safetensors",
    ):
        (checkpoint / name).write_text("x")
    (checkpoint.parent / "fit.json").write_text("{}")

    work(tmp_path, PLAN)
    back = sorted(str(f.relative_to(tmp_path / "w")) for f in (tmp_path / "w" / "fits").rglob("*"))

    assert back == [
        "fits/f1",
        "fits/f1/checkpoint-8000",
        "fits/f1/checkpoint-8000/adapter_config.json",
        f"fits/f1/checkpoint-8000/{ADAPTERS}",
        "fits/f1/fit.json",
    ]
    # The archive's own exclusions would hold a full model back even if it were copied.
    assert files_of(checkpoint.parent.parent.parent, ["fits"], WEIGHTS) == [
        "fits/f1/checkpoint-8000/adapter_config.json",
        f"fits/f1/checkpoint-8000/{ADAPTERS}",
        "fits/f1/fit.json",
    ]


TWO = AdaptationPlan(
    name="n",
    commit=COMMIT,
    inputs=INPUTS,
    per_card=2,
    fits=("f1 fit --out @F@/fits/f1", "f2 fit --out @F@/fits/f2"),
    answers=(
        "a answer --fits @F@/fits/f1 @F@/fits/f2 @IN@/fits/f0 --checkpoint c --out @W@/a.csv",
    ),
)


def test_an_answer_of_several_fits_waits_on_each_and_follows_any_redone(tmp_path: Path) -> None:
    first = work(tmp_path, TWO, failing="f2")
    again = work(tmp_path, TWO)
    with (tmp_path / "w" / "results.txt").open("a") as results:
        results.write("[f2] exit 0 in 1 s\n")
    after_refit = work(tmp_path, TWO)

    # Its second fit failed: the answer waits, though the first ended well.
    assert sorted(c.split()[0] for c in first) == ["fit", "fit"]
    assert [c.split()[0] for c in again] == ["fit", "answer"]
    # The second fit was redone after the answer: the answer is redone too.
    assert [c.split()[0] for c in after_refit] == ["answer"]
    assert work(tmp_path, TWO) == []
