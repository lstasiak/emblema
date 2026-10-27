"""The Mars Express audit measures each calendar month of the wide power and angle tables.

Held to small files in the publisher's layout whose months, channels and readings are known.
"""

from pathlib import Path

import numpy as np

from scripts.mex_power_audit import measure, months_of

JANUARY_2010 = 1_262_304_000_000  # 2010-01-01T00:00Z in milliseconds since the epoch
HOUR = 3_600_000


def test_an_instant_falls_in_its_utc_calendar_month() -> None:
    stamps = np.asarray(
        [JANUARY_2010 - 1, JANUARY_2010, JANUARY_2010 + 31 * 24 * HOUR], dtype=np.int64
    )

    assert months_of(stamps).tolist() == ["2009-12", "2010-01", "2010-02"]


def test_each_month_is_a_unit_and_an_empty_cell_no_reading(tmp_path: Path) -> None:
    train = tmp_path / "train_set"
    train.mkdir()
    power = ["ut_ms,NPWD1,NPWD2"]
    power += [f"{JANUARY_2010 + h * HOUR},1.0,{'' if h == 1 else '2.0'}" for h in range(4)]
    power += [f"{JANUARY_2010 + 31 * 24 * HOUR + h * HOUR},1.0,2.0" for h in range(2)]
    (train / "power--2010-01-01_2010-03-01.csv").write_text("\n".join(power) + "\n")
    angles = ["ut_ms,sa", f"{JANUARY_2010 + HOUR // 2},5.0", f"{JANUARY_2010 + 3 * HOUR // 2},6.0"]
    (train / "context--2010-01-01_2010-03-01--saaf.csv").write_text("\n".join(angles) + "\n")
    (tmp_path / "test_set").mkdir()
    (tmp_path / "test_set" / "power--2014-04-14_2016-03-01.csv").write_text("ut_ms,NPWD1\n1,1.0\n")

    january, february = measure(tmp_path)

    assert (january.unit, january.group, january.channels, january.observations) == (
        "mex/2010-01",
        "mex",
        3,
        4 + 3 + 2,
    )
    assert january.instants == 6
    assert january.channel_share == 1.0
    assert (february.unit, february.channels, february.observations) == ("mex/2010-02", 2, 4)
