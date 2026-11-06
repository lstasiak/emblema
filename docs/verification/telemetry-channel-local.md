# Each channel against its own recent hours on the first mission

The events that every readout of the first mission misses are local by the benchmark's
annotation, touch a median of 3 of the 48 channels and last a median of 5.5 hours; the found
ones touch 12 channels and last 15 ([telemetry-context.md](telemetry-context.md)). A detector
over the mean of every channel's cells sees little of such an event, and a longer window
averaged the same way saw less. This note reads each channel's hour against that channel's own
previous hours, with no model first: the departure of the hour's summary from the median of the
hours before it, over their spread, floored; a window's score the largest departure over its
channels. Beside it, the same departures taught every event, alone and beside the encoder's
state of the hour, so the note says whether the hours before carry what the hour's state does
not.

## 2026-11-06 — declared before the run: departures over 6, 24 and 72 hours, chosen on later fitting months, read beside the states' direction and the cells' bar

**Questions.** (1) Taught every event of the fitting side, does Fisher's direction over the
hour's state beside the channels' departures read the validation months above the direction
over the state alone, 0.275 on the same windows? (2) Without a label, does the largest
departure over the channels read above the cells' best zero-label bar, the farthest feature,
refitted here under the floored standardisation? Beside them, descriptively: the departures
taught alone, under the direction and under trees; the mean in place of the largest; which
events each arm finds at 1 % of normal windows, the local ones and the 11 unfound before among
them; and how many of each arm's top false alarms follow an event.

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/telemetry_channel_local_report.py`, at the commit of this declaration; the standardisation floored at `ad41a091` |
| Machine | M1 Pro, the CPU; validation numbers |
| Task, sides, cut | as in the context report: `esa-mission1-events`, the fitting side 2000-01 to 2006-12 (61,324 windows, 99 events), 56 validation months (41,088 windows, 62 events); the choice fits the 48 months before 2004-01 and chooses on the 36 from it (70 events taught); the refit fits the whole side and reads the validation months; a window whose context of 72 hours reaches into a month the task never loaded is not read under any arm (1,704 windows, one event wholly) |
| Cells and states | the context report's published one-hour cells, laid once (`cells-1h.npy`, 59 rows of the corpus's channels), and its states of the untrained encoder at seeds 1, 2, 3 over the one-hour layout; nothing is embedded anew |
| Departure | each channel's hour as five values, the mean of its bins' means, the least minimum, the largest maximum, the last reading and the log of the readings counted (`hour`), or its six bins' values (`bins`); each value's distance from the median of the same channel's values over the previous H − 1 hours, over their scaled median absolute deviation floored at a tenth of the channel's spread on the fitting side (the hour summary's spread, for both feature sets); the largest over the values; nought where the channel is absent from the hour or fewer than half the previous hours are loaded |
| Arms | `hour` and `bins` by H ∈ {6, 24, 72}: six arms, each under every detector; the arm chosen on the selection months per family and detector |
| Detectors | with no label: the largest departure over the channels (`max`) and the mean over those present (`mean`), each score against its month as the campaigns' alarm reads it; taught the events: Fisher's direction over the 59 departures (`direction`), over the hour's state beside them (`combined`, per seed of the state), and trees over the departures (`trees`, per seed); the recipes are the campaigns' |
| Controls | the direction and the whitened distance over the states, per seed, and the cells' farthest feature, refitted here on the same windows and under the floored standardisation |
| Seeds | 1, 2, 3 draw the states and the trees; `max`, `mean` and `direction` draw nothing and stand beside every seed of a seeded control |
| Reading | area and recall at 1 % of normal windows, each month's scores against that month; 10,000 resamples over months, seeds paired where both sides have them; the validation months stay shut until the choice is made |
| Command | `data/report/l38/local/measure.sh select`, `choose.sh`, `measure.sh refit --controls`, `check.sh`, `read.sh` |
| Cost | from the trial, on the CPU: the hours' summaries of the 102,412 loaded windows once, then the context's median and spread per arm, kept and shared by both sections: 25 s at `hour-6h`, 209 s at `hour-72h`, 1,644 s at `bins-72h` (the bins' context holds 426 values a channel); the floored departures of a section seconds; every detector seconds (the largest and the mean under a second, the direction 2 to 3 s, the trees 3 to 6 s, the farthest feature 7 to 14 s). The whole run is under an hour |

**The rule, stated before the run.**

- *The choice*, for each family and detector from the selection months alone: the arm with the
  highest mean area over its seeds; of every arm within 0.01 of it the shorter span, then
  `hour` before `bins`.
- *The reading*, two readings declared as one family under Holm, each the chosen arm less its
  control on the validation months with its interval over months: (1) `combined` less the
  direction over the states, seeds paired, T = 2·SE over the three seeds; (2) `max` less the
  cells' farthest feature, neither side drawing, so T is nought and the interval over months
  decides. The least gain that counts is the detection tasks' registered 0.05. Everything beside
  is read the same way, descriptively; the direction over the departures alone and the trees
  against the states' direction, `mean` against the bar.
- *The check*, before any area is read: the states' direction refitted here must rank the
  context report's refit cells' windows as they did under every seed, Spearman's correlation
  at least 0.9999 and the top 1 % of windows held whole; the floor on the standardisation moves
  a score by a little, so the check reads ranks, not gaps. Below that the run is stopped and
  the code is wrong.
- *The events*, descriptively: the events each refit cell finds at 1 % of normal windows,
  the local ones among them, and the 11 unfound before; those 11 were read when the
  local few-channel profile was read off them, so they decide nothing.

| Outcome | Condition | What follows |
| --- | --- | --- |
| Information: clearly above | (1) at least +0.05 and T, interval above zero, under Holm | the hours before carry what the hour's state does not: the pretext produces states per channel against the channel's own history, not one state of the window |
| Information: around or below | (1) inside ±T, or below | channel-local context holds nothing the taught direction lacks; 0.26 is read as a property of the cells' input under linear readouts, and the leg's aims are talked through before any GPU |
| Product: clearly above | (2) at least +0.05, interval above zero, under Holm | a detector with no model and no label enters the package as the leg's zero-label bar |
| Product: around or below | (2) inside ±0.05 or below | the largest departure over the channels is no better a zero-label bar than the farthest feature; the bar stays |
| Undecided | (1) with T above 0.05 | the seeds spread more than the least gain; whether more seeds are worth their hour is decided with the size read |

**Power, stated before the run.** The direction over the states spreads 0.01 to 0.02 over the
seeds on the refit (0.285, 0.282, 0.257), so T under (1) is about 0.03 and a true gain of +0.05
is read above zero with probability about 0.8; the interval over months of a paired direction
reading had a half-width of 0.04 to 0.06. Under (2) neither side draws; the interval over
months of the context report's farthest arm had a half-width of 0.09, so a gain under 0.09 may
hold zero, and "around" under (2) leaves a gain of that size open.

**Size from above.** The 11 unfound events are 18 % of the 62; found at the rank the found
ones have, they would raise the event-weighted area by about 0.1 under (1). Under (2) the bar
read 0.02 on these windows before the floor and about 0.2 under one alignment of the months, so
where it stands is itself read here.

**Prediction.** Under (1) the difference lies in [−0.03; +0.08], near +0.02: the direction is
taught the same events and the departures add 59 dimensions it can weigh or ignore; the gain
is bounded by the local events' share. Under (2) `max` lies in [−0.05; +0.15] against the bar,
near +0.03: a departure in any of 59 channels raises an alarm, and the bar's number after the
floor is not known. `hour` is chosen over `bins` under `max`, since one bin departs more often
by chance, and 24 hours over 6, since a median of five hours is thin. The direction over the
departures alone reads below the states' direction; the trees over the departures above the
trees over the cells. The `max` arm's top false alarms follow events more often than the
states' direction's, since the hours after a long event depart from a median the event set.

**Trial, before this declaration.** Every command ran end to end at `hour-6h` and seed 1 on the tree before this declaration's commit, the tables written and deleted unopened; only the cell names, their seconds, the count of windows not read (120 of 41,088 at a widest context of 6 hours) and the checks were read. The direction over the states refitted here ranks the context report's 41,088 windows with Spearman 0.99998 and holds its top 1 % whole; the whitened distance ranks them the same; the cells' farthest feature, the bar, ranks them with Spearman 0.981 and holds 0.33 of its top 1 %, which is the floor's work on the near-constant features. The check's criterion is the one the rule states; its gap-based form from the context report read a largest gap of 4.1 and a mean of 0.026 here, the floor's work, and was replaced before this declaration. The check of the seeds 2 and 3 and of the whole grid comes with the run. On the selection months, read before for the choice, the controls' areas against the context report's: the direction 0.219, 0.203, 0.196 against 0.220, 0.203, 0.196; the whitened distance the same to the third decimal; the farthest feature 0.175 against 0.167. The widest arms were timed on the trial's tree, their departures unread.

**Limitations.** No model is trained; the taught arms are linear or trees over 59 numbers. The
departure reads each channel alone, so an event that moves several channels together within
each one's range is not its target. The spread is read over the previous hours as the data has
them, gaps included, and the floor is one share for every channel. Three seeds where a side
draws; the local arms draw nothing, so their intervals come from the months alone. The
validation months have been read before; the test months stay frozen.
