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

**Added beside, after the choice and before the reading (2026-11-06).** On the selection
months, which the choice reads, the largest departure over the channels read 0.170 to 0.179
under every arm with no label, while the direction over the 59 departures taught 70 events
read 0.001 to 0.031 and beside the state 0.001 to 0.054, below the state alone at 0.206. The
check that this is the representation and not the instrument, on those months only: a plain
logistic direction and a plain linear discriminant over the logged departures read 0.014 and
0.026, and the same two over the three largest departures of each window in order read 0.168,
the largest departure's own number. The events lie in which channel departs most, an order
statistic no weight per channel expresses. Two arms are added beside, under every detector's
rules and read descriptively: the direction over the three largest logged departures (`top`),
and over the hour's state beside them (`ordered`, per seed). The declared readings stand as
declared; the first reading of the validation months, written by the run before this
addition, was deleted unread, and the cells of this addition were measured under the
declaration's next commit.

**Limitations.** No model is trained; the taught arms are linear or trees over 59 numbers. The
departure reads each channel alone, so an event that moves several channels together within
each one's range is not its target. The spread is read over the previous hours as the data has
them, gaps included, and the floor is one share for every channel. Three seeds where a side
draws; the local arms draw nothing, so their intervals come from the months alone. The
validation months have been read before; the test months stay frozen.

## 2026-11-06 — measured: the largest departure reads the bar with no label and finds the events the taught direction finds; taught linearly the departures add nothing to the state, taught by trees they find 50 of 61 events at 1 %

**Questions.** As declared: (1) whether the hour's state beside the channels' departures,
taught every event, reads above the state alone under Fisher's direction; (2) whether the
largest departure over the channels, with no label, reads above the cells' farthest feature.

**Conditions.** As declared, at commits `b0c9f84f` (code), `7afeadae` (declaration) and
`161d068c` (the order-statistic arms added beside before the reading); `run.sh` on the M1 Pro's
CPU. The first run took 40 minutes, 37 of them the context statistics of the six arms (25 s at
`hour-6h` to 1,361 s at `bins-72h`); its reading was deleted unread and the second run, the
statistics kept, took 5 minutes: every cell under a second to 6 s, the farthest feature 9 s.
The check held under every seed: the states' direction refitted here ranks the context
report's windows with Spearman 0.99998, 0.99998 and 0.99995 and holds the top 1 % whole. Not
read: 1,704 of 41,088 validation windows, under every arm alike; 61 events read.

**The choice, on the months 2004 to 2006, 70 events taught; mean area over the seeds.**

| Family | Detector | hour-6h | bins-6h | hour-24h | bins-24h | hour-72h | bins-72h | Chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local | max | **0.170** | 0.161 | 0.163 | 0.171 | 0.177 | 0.179 | hour-6h |
| local | mean | **0.169** | 0.096 | 0.144 | 0.058 | 0.146 | 0.066 | hour-6h |
| taught | combined | 0.010 | **0.054** | 0.002 | 0.001 | 0.001 | 0.001 | bins-6h |
| taught | direction | 0.001 | **0.031** | 0.001 | 0.002 | 0.001 | 0.001 | bins-6h |
| taught | ordered | **0.203** | 0.198 | 0.207 | 0.197 | 0.206 | 0.198 | hour-6h |
| taught | top | 0.090 | 0.007 | 0.041 | **0.145** | 0.029 | 0.095 | bins-24h |
| taught | trees | 0.043 | **0.055** | 0.021 | 0.034 | 0.014 | 0.030 | bins-6h |

The largest departure reads alike under every span and both feature sets, within 0.02; the
shortest is chosen. On these months the states' direction reads 0.206 and the farthest
feature 0.175.

**The reading, on the validation months, 99 events taught; the chosen arm less its control,
10,000 resamples over months; T = 2·SE over the seeds where a side draws.**

| Family | Detector | Arm | Area per seed | Area | Recall @1 % | Control | Difference [interval] | p | T | Outcome | Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taught | combined | bins-6h, declared | 0.193 0.204 0.194 | 0.197 | 0.596 | 0.276 | **−0.079** [−0.138; −0.036] | 0.000 | 0.017 | below | passes |
| local | max | hour-6h, declared | 0.179 | 0.179 | 0.607 | 0.160 | **+0.019** [−0.093; +0.086] | 0.728 | 0 | around | does not pass |
| taught | ordered | hour-6h | 0.283 0.284 0.256 | 0.274 | 0.590 | 0.276 | −0.001 [−0.011; +0.003] | 0.500 | 0.004 | around | |
| taught | ordered | bins-24h, beside | 0.282 0.283 0.245 | 0.270 | 0.596 | 0.276 | −0.006 [−0.022; +0.006] | 0.308 | 0.009 | beside | |
| taught | direction | bins-6h | 0.189 | 0.189 | 0.607 | 0.276 | −0.087 [−0.150; −0.041] | 0.000 | 0.018 | below | |
| taught | top | bins-24h | 0.176 | 0.176 | 0.492 | 0.276 | −0.099 [−0.172; −0.045] | 0.000 | 0.018 | below | |
| taught | trees | bins-6h | 0.170 0.161 0.178 | 0.170 | **0.836** | 0.276 | −0.106 [−0.171; −0.005] | 0.040 | 0.026 | below | |
| taught | trees | bins-24h, beside | 0.209 0.189 0.163 | 0.187 | **0.880** | 0.276 | −0.089 [−0.153; −0.000] | 0.050 | 0.011 | beside | |
| local | mean | hour-6h | 0.188 | 0.188 | 0.672 | 0.160 | +0.027 [−0.079; +0.095] | 0.551 | 0 | around | |

The controls: the states' direction 0.287, 0.282, 0.259 (recall 0.596 at 1 %), the whitened
distance 0.097, 0.095, 0.096 (0.180), the farthest feature 0.160 (0.426), under the floored
standardisation. The 12 rows not shown read −0.27 to +0.03.

**Events found at 1 % of normal windows, of the 61 read; the local ones among them; the 11
no arm found before; the share of the top false alarms within 6 hours after an event.**

| Cell | Found | Local | Of the 11 | After 6 h |
| --- | --- | --- | --- | --- |
| states' direction, seeds 1 to 3 | 39, 30, 40 | 15, 9, 17 | 0, 0, 3 | 0.01 to 0.02 |
| states' whitened distance | 11 | 8 | 0 | 0.00 |
| cells' farthest feature | 26 | 8 | 0 | 0.02 |
| local max, hour-6h | 37 | 12 | 3 (110, 132, 166) | 0.00 |
| local mean, hour-6h | 41 | 17 | 3 (132, 166, 184) | 0.01 |
| taught ordered, hour-6h, seeds 1 to 3 | 40, 29, 39 | 16, 8, 16 | 0, 0, 2 | 0.01 to 0.02 |
| taught trees, hour-6h, seeds 1 to 3 | 50, 52, 51 | 25, 25, 24 | 7, 7, 6 | 0.03 |
| taught trees, bins-6h, seeds 1 to 3 | 51, 50, 52 | 25, 24, 26 | 8, 7, 7 | 0.05 to 0.06 |
| taught trees, bins-24h, seeds 1 to 3 | 54, 53, 54 | 27, 26, 27 | 7, 7, 7 | 0.14 to 0.16 |

The trees over the departures find the events 50, 110, 132, 156, 157, 166 and 184 under every
seed and arm, and 136 under some; the trees over the cells found 27 to 31 events and none of
the 11 ([telemetry-context.md](telemetry-context.md)).

**Conclusions.**

- **(1) Below, under Holm: taught linearly, the departures beside the state read below the
  state**, −0.079 [−0.138; −0.036], and the direction over the 59 departures alone reads
  0.19 against 0.28. Read as order statistics beside the state (`ordered`) they read the
  state's number, −0.001 [−0.011; +0.003]: a linear readout taught the events takes nothing
  from channel-local context that the hour's state does not already hold.
- **(2) Around: with no label, the largest departure over the channels reads 0.179 against
  the bar's 0.160**, +0.019 [−0.093; +0.086], and 0.096 for the whitened distance over the
  states. It finds 37 of the 61 events at 1 % of normal windows, as many as the taught
  direction (30 to 40), three of the 11 unfound before among them, and its top false alarms do
  not trail the events. As a zero-label detector it is level with the best bar and twice the
  backbone's reading; it does not pass the least gain over the bar.
- **Beside, and not declared: taught by trees, the departures find 50 to 54 of the 61 events
  at 1 % of normal windows**, against 30 to 40 for the taught direction over the states and
  27 to 31 for the trees over the cells, seven of the 11 unfound before among them under every
  seed. Their area stays at 0.17 to 0.19, since an event's later windows rank low once its
  context holds the event and the top false alarms follow events (0.14 at 24 hours of
  context); by the registered area they read below the direction. The information the
  one-hour readouts lack is in the channels' departures from their own recent hours, and it
  is read by which channels depart, not by how much on average.
- **The predictions.** (1) [−0.03; +0.08] missed below; (2) [−0.05; +0.15] held, near its
  centre. `hour` was chosen over `bins` and 6 hours over 24 under the largest departure, the
  second against the prediction: the spans read within 0.02 of one another.
- **What follows for the pretext.** The declared row for (1) says the leg's aims are talked
  through before any GPU, and the trees say what to talk about: a representation per channel
  against its own history carries the missed events, and a readout that asks which channels
  depart reads it. A pretext that produces one state of the window averaged over the channels
  aims at the wrong thing; one that produces a state per channel against the channel's own
  recent hours, read by the largest few, aims at what the trees found.

**Limitations.** The departure reads each channel alone against a median of five to 71
hours; the 11 channels the mission never fills depart nought. The floor is the hour summary's
spread for both feature sets. The trees are read beside, not declared, and their recall at 1 %
is not the registered area; an event found by its onset and lost in its tail counts once in
the recall and little in the area. The order-statistic arms were added after the choice and
before the reading, on the selection months' evidence, and stand beside. Three seeds where a
side draws. The validation months have been read before; the test months stay frozen.

## 2026-11-06 — declared before the run: the second mission, as a confirmation on data the hypothesis did not come from

**Questions.** The same two as on the first mission, on the second: (1) whether the hour's
state beside the channels' departures, taught every event, reads above the state alone under
Fisher's direction; (2) whether the largest departure over the channels, with no label, reads
above the cells' farthest feature. Beside, descriptively, the same arms, and in particular
whether the trees over the departures find more events at 1 % of normal windows than the
taught direction over the states, as they did on the first mission. The local few-channel
profile was read off the first mission's validation events, so the first mission's reading
confirms nothing of itself; the second mission's events were never read for it.

**Conditions.** As on the first mission, with these differences.

| | |
| --- | --- |
| Code | `scripts/telemetry_channel_local_report.py` at the commit of this declaration; the context report lays a month published 16 s off the hour on the hourly grid (`TOLERANCE_HOURS`); the check reads ranks by unit |
| Task | `esa-mission2-events` `09368ab9…`, the same corpus manifest: the fitting side 2000-01 to 2001-09, 21 months, 15,335 windows, 272 events; the validation months 2001-10 to 2003-06, 14 of them, 10,223 windows, 236 events in 475 windows, so an event lasts two windows at the median; the corpus reads 11 of the mission's channels, so 48 of the 59 rows are absent and depart nought; of the mission's 622 annotated events 45 are local |
| Cut for the choice | the 14 months before 2001-03 fit, the 7 from it choose |
| Cells and states | laid and embedded by the context report's reader once (32 s and 6 to 7 s a seed on MPS), the same layout and encoder shape |
| Not read | a validation window whose context of 72 hours reaches into a month the task never loaded: 355 of 10,223, touching 7 events |
| Check | the refitted whitened distance over the states against the readout screen's second-mission cells (`untrained_knn@distance=whitened`, seeds 1 to 3), which read no unit alone: Spearman's correlation at least 0.9999 and the top 1 % held whole within every unit, the least over the 14 units |
| Command | `data/report/l38/local-m2/run.sh` |
| Cost | from the trial: the departures of `hour-6h` 14 s, every cell under 3 s; the widest arm's statistics a few minutes; the whole run about a quarter of an hour |

**The rule** is the first mission's, unchanged: the choice per family and detector on the
selection months; the two declared readings under Holm with T = 2·SE over the seeds where a
side draws and nought where neither does; the least gain 0.05; the rest descriptive. A
confirmation reads the two missions together: a recommendation of the first mission's reading
stands where the second mission's reading has the same sign and holds its prediction, and is
scoped to the first mission's kind of events where it does not.

**Power.** Fourteen months make the interval over months about twice as wide as on the first
mission; with 236 events of two windows and about 97 normal windows flagged at 1 %, the recall
at 1 % is bounded near 0.4 for any arm, so the trees' lead is read in the area as much as in
the recall here.

**Prediction, from the first mission's intervals.** (1) in [−0.14; +0.03], near −0.06, below
or around. (2) in [−0.09; +0.09], near +0.02, around; the largest departure reads above the
whitened distance over the states if that reads as low as it did on the first mission, and
not if the second mission's global rare events lift it, as the readout screen saw. Order
statistics beside the state in [−0.03; +0.02]. The trees over the departures against the
direction over the states: in the recall at 1 % [−0.05; +0.25], near +0.10; in the area
[−0.20; +0.05]. The arm chosen under the largest departure reads within 0.02 of the others.

**Trial, before this declaration.** Every command ran end to end at `hour-6h` and seed 1, the
tables deleted unopened; only the cell names, their seconds, the count of windows not read
(25 of 10,223 at a widest context of 6 hours) and the check were read. The check held: the
whitened distance refitted here ranks the screen's windows within every unit with Spearman
1.00000 and holds every unit's top 1 %, so the states are the screen's.

**Limitations.** The second mission's events are mostly global rare events, so it confirms the
detectors, not the local few-channel mechanism; its 45 local events are read in the events
table. Eleven channels. Fourteen months.

## 2026-11-06 — measured on the second mission: with no label the largest departure reads 0.885 against the bar's 0.627, level with the taught direction; taught linearly the departures again add nothing

**Questions.** As declared for the second mission.

**Conditions.** As declared, at commit `8488e834` (the tolerance `930bae85`, the by-unit check
`64df4981`); `run.sh` on the M1 Pro, 10 minutes in all, 8 of them the six arms' context statistics (14 s at `hour-6h` to 286 s at `bins-72h`).
The check held under every seed: the refitted whitened distance ranks the readout screen's
second-mission windows within every unit with Spearman 1.00000 and holds every unit's top 1 %.
Not read: 355 of 10,223 windows under every arm alike; 229 events read, 4 of them local.

**The choice, on the months 2001-03 to 2001-09; mean area over the seeds.**

| Family | Detector | hour-6h | bins-6h | hour-24h | bins-24h | hour-72h | bins-72h | Chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local | max | 0.920 | 0.925 | 0.906 | 0.889 | **0.932** | 0.915 | bins-6h |
| local | mean | **0.898** | 0.889 | 0.827 | 0.815 | 0.850 | 0.820 | hour-6h |
| taught | combined | 0.925 | 0.924 | **0.928** | 0.921 | 0.927 | 0.922 | hour-6h |
| taught | direction | 0.919 | 0.921 | **0.922** | 0.914 | 0.909 | 0.903 | hour-6h |
| taught | ordered | 0.934 | **0.936** | 0.933 | 0.934 | 0.933 | 0.935 | hour-6h |
| taught | top | **0.871** | 0.861 | 0.842 | 0.841 | 0.783 | 0.771 | hour-6h |
| taught | trees | **0.887** | 0.785 | 0.880 | 0.702 | 0.876 | 0.666 | hour-6h |

**The reading, on the 14 validation months, 272 events taught.**

| Family | Detector | Arm | Area per seed | Area | Recall @1 % | Control | Difference [interval] | p | T | Outcome | Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local | max | bins-6h, declared | 0.885 | 0.885 | 0.996 | 0.627 | **+0.258** [+0.199; +0.322] | 0.000 | 0 | clearly above | passes |
| taught | combined | hour-6h, declared | 0.847 0.845 0.853 | 0.848 | 0.978 | 0.900 | **−0.052** [−0.079; −0.028] | 0.000 | 0.010 | below | passes |
| local | mean | hour-6h | 0.873 | 0.873 | 0.991 | 0.627 | +0.246 [+0.187; +0.313] | 0.000 | 0 | clearly above | |
| taught | ordered | hour-6h | 0.894 0.895 0.915 | 0.901 | 0.996 | 0.900 | +0.001 [+0.001; +0.002] | 0.000 | 0.000 | around | |
| taught | direction | hour-6h | 0.828 | 0.828 | 0.974 | 0.900 | −0.072 [−0.102; −0.039] | 0.000 | 0.014 | below | |
| taught | top | hour-6h | 0.862 | 0.862 | 0.987 | 0.900 | −0.037 [−0.060; −0.011] | 0.007 | 0.014 | below | |
| taught | trees | hour-6h | 0.825 0.817 0.822 | 0.821 | 0.977 | 0.900 | −0.078 [−0.125; −0.024] | 0.001 | 0.014 | below | |
| taught | trees | bins-6h, beside | 0.907 0.904 0.906 | 0.905 | 0.996 | 0.900 | +0.006 [−0.017; +0.026] | 0.683 | 0.014 | beside | |

The controls: the states' direction 0.892, 0.894, 0.914 (recall 0.996), the whitened distance
0.657, 0.693, 0.710 (0.889), the farthest feature 0.627 (0.869). At 1 % of normal windows
the largest departure, the taught direction, the order statistics and the trees each find 228
of the 229 events, the four local ones among them, and miss the same one; the whitened
distance finds 197 and the farthest feature 199. The column of events unfound before and the
share of false alarms after events say nothing here: the first names the first mission's
events, and with 236 events in 14 months nearly every normal window lies within 72 hours of
one.

**Conclusions, the two missions together.**

- **(2) holds and strengthens: with no label, each channel against its own recent hours is a
  detector level with the best bar on the first mission and far above it on the second**,
  +0.258 [+0.199; +0.322] under Holm, 0.885 against 0.627 and against 0.687 for the whitened
  distance over the states; it reaches the taught direction's recall at 1 % on both missions
  (0.61 against 0.60; 0.996 against 0.996) and its area on the second (0.885 against 0.900).
  The prediction [−0.09; +0.09] missed above: the bar is weak on the second mission, the
  departure is not.
- **(1) holds: taught linearly, the departures add nothing to the state on either mission**,
  −0.052 [−0.079; −0.028] under Holm here, −0.079 there; as order statistics beside the state
  +0.001 [+0.001; +0.002] here, −0.001 there. The prediction [−0.14; +0.03] held.
- **The trees' lead is the first mission's.** There they found 50 to 54 of 61 events at 1 %
  against 30 to 40 for the taught direction; here every taught arm and the largest departure
  find 228 of 229, and the trees read the direction's area (+0.006 [−0.017; +0.026] at
  `bins-6h`). The second mission's events are global rare events the hour's state already
  shows; the recommendation that reads "which channels depart" is scoped to events like the
  first mission's local few-channel ones, as the declared confirmation rule says.
- **A wrong statement in the declaration.** Its power paragraph said the recall at 1 % was
  bounded near 0.4 here by the 97 normal windows flagged; the share counts normal windows
  only, so every event window may rank above it, and the recall reached 0.996. The bound
  played no part in the rule.
- **What follows.** The product ticket for a channel-local detector with no label stands on
  two missions. The pretext's shape, a state per channel against the channel's own history
  read by the largest few, stands as the target for the events the hour's state misses, which
  the first mission has and the second does not; a pretext is marked against the departures
  themselves.

**Limitations.** Eleven channels and 14 months; 4 local events among 229. The taught arms
read near a ceiling (0.90 to 0.94 on the selection months), so differences among them are
small by construction. The spans read within 0.04 of one another under the largest
departure, wider than the predicted 0.02. The validation months have been read before by the
readout screen; the test months stay frozen.

## 2026-11-06 — measured: the package's detector reads both missions as the report did

### Question

The report above read the channel-local detector with no label on both missions. Does the
package's detector (ADR-0075), fitted by a campaign cell, rank every month's windows as the
report did, under the acceptance declared in the ADR before the cells ran?

### Conditions

| | |
|---|---|
| Code | `08f98fa9` (detector), cells ordered from `820a41e2`; rank check fixed in `a1391d1a` |
| Machine | M1 Pro, CPU, campaign cell by order on the host |
| Campaigns | `telemetry-mission1-local` (`channel_local@reference=unit`, the report's `hour-6h`), `telemetry-mission2-local` (`channel_local@departure=bins,reference=unit`, the report's `bins-6h`), one seed, the cells' farthest feature as control |
| Against | the report's refit answers for the same arm: `refit-local-max-hour-6h` (first mission), `refit-local-max-bins-6h` (second) |
| Acceptance | within every month Spearman's coefficient at least 0.9999 and the report's top 1 % held whole |
| Commands | `data/report/l44/run.sh m2`, `m1`; `data/report/l44/check.sh m2`, `m1` (`telemetry_channel_local_report.py check --by-unit`) |

### Results

| | Second mission | First mission |
|---|---|---|
| Detector's cell, seconds once the block is at hand | 58 s | 249 s |
| Both cells' run, wall time / peak memory | 114 s / 3.6 GB | 511 s / 11.0 GB |
| Fitting windows: cell / report | 15,335 / 15,335 | 61,363 / 61,324 |
| Scored windows in common | 10,223 | 41,088 |
| Answers identical | all | none; median difference within a month −1.7·10⁻⁷ |
| Least Spearman over the months | 1.00000 | 0.99998 |
| Top 1 % held | 1.000 | 1.000 |
| Area / recall at 1 %, report's read windows: cell | 0.8845 / 0.9956 | 0.1790 / 0.6066 |
| The same, report | 0.8845 / 0.9956 | 0.1790 / 0.6066 |

The report dropped from its fit the 39 first-mission windows that overlap a downlink gap, which
it learned from the ground truth; the cell fits on every fitting window, as every campaign
detector does. The floor of the channels' spreads moves with them, so the first mission's
answers shift by the alarm's level and, where the floor binds, by more: up to 0.17 in one month
(2007-04). The second mission has no such window and matches bit for bit.

The first check of the first mission read 0.99941 in 2013-03. The check ranked each scoring by
the order its sort left, and 477 of that month's 744 windows tie, since most hours depart by
nothing; the same answers read 1.00000 under a stable sort and 0.999999 under mean ranks,
Spearman's coefficient as declared. The check now gives tied windows their mean rank
(`a1391d1a`), as does the count of windows moved more than ten ranks (`32b4b11b`: 1,439 under
the sort's order, 28 under mean ranks, at a median difference of 1.6·10⁻⁴). The threshold is
unchanged.

### Conclusions

- The package's detector is the report's, within the declared acceptance on both missions.
- On the second mission it is the same to the bit; on the first the only difference is the
  fitting side the cell is handed, and it moves no area or recall at four decimals.
- The detector's cell takes about four minutes on the host for the first mission and one for
  the second; the cells' bar beside it reads 0.1544 on the first, as the readout screen did.
- The detector is now a bar a backbone is held to on the telemetry tasks, read by area and by
  the events found at 1 % false alarms.

### Limitations

- One seed: neither detector draws anything, so seeds would repeat the same answers.
- The run's peak memory on the first mission (11 GB) holds both cells' inputs; a longer history
  over the spans would grow the detector's share.
- Tasks whose units each keep their own time axis are refused, not read.

## 2026-11-10 — declared before the run: the departure bar chosen by events found above chance, on months no arm was read on

**Question.** Which no-model arm is the first mission's channel-local bar when the choice uses
the measure the verdict is read by? The bar `max`, `hour-6h` was chosen by mean area on the
selection months. From 2026-11-09 the first mission's verdict reads the events found at 1 % of
normal windows (`telemetry-ceiling.md`).

**Why not the selection months.** This choice was first written for the selection months (2004
to 2006). Before it was committed, a check of how the arms' alarms are spread printed how many
windows each arm's real scores flag there, event windows included, which less the 1 % of normal
windows counts the event windows each arm flags. The ceiling note had published that the twelve
arms together find 22 of those 29 events, and its run holds each arm's count. The rule below was
shaped after that print, so a choice there would not be blind. The choice moves to the fitting
side's months before them, 2000 to 2003, which no arm has been read on; the selection months are
read beside, marked as not blind. Every check of scores against events on the months 2000 to
2003 ran on rotated scores.

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/telemetry_channel_local_report.py` at the commit of this declaration; the measure at `51864599`, the ground truth's overlaps at `cfd34c29` |
| Machine | M1 Pro, the CPU; no model; the measure 18 minutes on its first, trial run and under a minute at the commit; the choice seconds |
| Side | the 12 no-model arms (`max`, `mean` by `hour`, `bins` by 6, 24, 72 hours), each score read against its month; scored: 2000-01 to 2003-12, 48 months, 35,020 windows less 71 whose 72 hours reach before the first month; the floor set on 2004-01 to 2006-12 (26,304 windows), where each cell's own alarm level is set too, which shifts every score of a cell alike and so does not move the alarm in the next row |
| Events | 70 by their windows' targets; `class_22` left out, every window touching one of its events set aside, neither event nor normal: its 28 events among the 70 and 4 more with no window of their own, 32 in all, and 35 windows, leaving 42 events in 34,914 windows |
| Alarm | the score of the normal window past 1 % of the side's normal windows, as the harness counts it; an event is found when one of its windows scores above it |
| Chance | each arm's events found, averaged over 1,000 rotations of every month's scores by an offset of at least one window, the same rotations for every arm (seed 1) |
| Commands | `data/report/l47/measure.sh data/report/l47/early`; `telemetry_channel_local_report.py choose-recall --dir data/report/l47/early --section early --task esa-mission1-events --leave-out class_22 --out data/report/l47/choice`; beside, the same without `--leave-out`, and with `--dir data/report/l38/local/run --section select` without `--leave-out` |

**Why `class_22` is left out.** Events by their windows' targets; length in windows:

| Months | Events | `class_22` | Most frequent classes | Median windows | Short (≤ 6 windows) |
| --- | --- | --- | --- | --- | --- |
| 2000 to 2003 | 70 | 28 | 22, 7, 14 | 5 | 0.54 |
| 2000 to 2003, less `class_22` | 42 | 0 | 7, 14 | 25.5 | 0.24 |
| 2004 to 2006 | 29 | 0 | 7, 3, 14 | 8 | 0.41 |
| validation | 61 | 1 | 7, 14, 3 | 10 | 0.44 |

With it, 40 % of the choice would rest on a class the months the bar is read on hardly hold.
Without it, the events left run longer than on those months. A window's target names only the
lowest event it overlaps, so five events touching these months carry no window of their own
number, four of them `class_22`; the windows are set aside by every event they overlap.

**The rule, stated before the run.**

- *The choice*: the arm with the largest excess, the events found less the events found by
  chance; then the most found; then the most found averaged over 0.5, 1, 1.5 and 2 % of normal
  windows; then the shorter span, `hour` before `bins`, `max` before `mean`.
- *Beside*, descriptively: the events found at 1 %, the averaged count, and the events found by
  their onset, by a window ending less than 6 hours after the end of the event's first scored
  window, at the same alarm and not corrected for chance; the short events' excess, on the 10
  events of at most 6 windows; the same with all 70 events; and the selection months by the
  same rule, not blind.
- *What follows.* The chosen arm becomes the first mission's channel-local bar in the
  registered rules, as `channel_local` with its knobs (`channels`, `departure`, `span`) and
  `reference=unit`. If it is not `max`, `hour-6h`, the package's detector is checked against the
  report on that arm, rank by rank within each month, before any campaign reads it. Two
  findings go back to a decision before the next reading with a telemetry verdict: if the
  chosen arm finds more events in all than `max`, `hour-6h` but fewer by their onset, its lead
  lies in events' later hours; and if its short events' excess is 3 or more below the largest
  among the arms, a third of the 10 (under the rotations two arms' short counts differ with a
  standard deviation of 0.3 to 0.7), its lead lies in long events, which these months
  over-represent. The second mission keeps `bins` with `max` and 6 hours: its bar found every
  event of its selection months (`telemetry-history-pretext.md`), so the count has no room to
  choose there.
- *Beside every telemetry reading from now on*: the excess over the same rotations, next to the
  events found, for the candidate and the bar alike.

**Why the excess and not the count.** On these months an arm finds by chance (the mean over
500 rotations, its standard deviation beside):

| Arm | `max` | `mean` |
| --- | --- | --- |
| `hour-6h` | 10.7 (1.8) | 14.1 (2.1) |
| `bins-6h` | 14.1 (1.9) | 13.5 (2.0) |
| `hour-24h` | 9.7 (1.7) | 11.1 (2.1) |
| `bins-24h` | 9.5 (1.8) | 9.7 (2.3) |
| `hour-72h` | 8.8 (2.2) | 6.4 (2.7) |
| `bins-72h` | 7.6 (1.7) | 7.2 (2.6) |

The arms differ by up to 7.7 events of 42, 0.18, by chance alone; of the 10 short events,
chance finds 0.1 to 0.3. On the selection months a random score per window found about 10 of
29 and the arms' chance ran from 3 to 11, the arms over 72 hours putting their alarms into
fewer, longer runs (18 to 92 against 158 to 206 under a rotation); a budget of false-alarm
runs did not even this out.

**What has been seen.** On 2000 to 2003: the events' classes, lengths and overlaps; that every
score is finite; no arm's scores against the events except under rotation. Elsewhere, among
others: on the selection months every arm's area, `channel_local`'s found share (0.586), other
detectors' shares, the twelve arms' union (22 of 29) and every arm's flagged windows; on the
validation months six arms' found shares (`max` 0.525 to 0.623, `mean` 0.656 to 0.689), which
events each finds, and their union, 48 of 61.

**Prediction.** A weak lean from point shares, with no interval. Under the rotations two arms'
counts differ with a standard deviation of 1.5 to 3.1 events (median 2.3), so arms within
about 5 events are chosen among by noise. Source: each validation share scaled to 42 events,
less the arm's chance here; the share carries the validation months' events and chance, which
differ from these in length and class by an amount that differs by arm, direction unknown.
Excess near 19 for `mean`, `bins-24h`, 17 for `max`, `bins-24h`, 15 for `max`, `hour-6h`, 14 for
`mean` over `hour-6h` and `bins-6h`, 8 for `max`, `bins-6h`; the other six arms have no
validation share. The lean: `bins-24h`, `mean` before `max`; the gap of about 4 events to
`max`, `hour-6h` is within the noise, so the bar may stand. No prediction for the onset or the
short events.

**Trial, before this declaration.** The measure of the 12 arms ran first under `--trial`, which
only waives the committed tree and reads no event; its answers are byte for byte those of the
run at `51864599`. The choice ran three times on rotated scores (`--trial 1`, `2`, `3`): the 12
arms over the same windows and events, the excess between −3.5 and +4.8 events and the short
events' between −0.3 and +0.8; 500 further rotations gave the spreads above. Two reviews of the
code found the leave-out reading only a window's target, which names the lowest overlapping
event, and a run without `--leave-out` writing over one with it; both were fixed before the
measure.

**Limitations.** Forty-two events and 48 months, the arms measured once, so the choice has no
interval. These are the mission's first years: drift, classes (`class_3` 2 events here, 14 on
the validation months) and lengths differ from the months the bar is read on, and 10 short
events check the last only coarsely. The onset is the first window the side scores. The rule was
written after the selection months' flagged windows were seen; applied to months not read, it
chooses blind, but its shape is not independent of that look.

## 2026-11-10 — measured: by the events found above chance, the mean departure over 72 hours; the 6-hour bar finds a third as many above chance

**Conditions.** As declared, at `471ba43f`: `choose-recall` on the months 2000 to 2003 less
`class_22`, 42 events, 1,000 rotations; beside, the same with all 70 events and the selection
months (not blind). Output in `data/report/l47/choice/`.

**The choice.** Events of 42; chance and excess in events; onset by the first 6 hours; short
events' excess of 10.

| Arm | Found | Chance | Excess | Onset | Short excess |
| --- | --- | --- | --- | --- | --- |
| **`mean`, `hour-72h`** (chosen) | 21 | 6.4 | **14.6** | 15 | **3.9** |
| `mean`, `bins-72h` | 21 | 7.2 | 13.8 | 16 | 3.8 |
| `mean`, `bins-24h` | 23 | 9.6 | 13.4 | 14 | 2.7 |
| `max`, `hour-24h` | 22 | 9.7 | 12.3 | 14 | 2.9 |
| `mean`, `hour-24h` | 23 | 11.1 | 11.9 | 16 | 3.8 |
| `mean`, `hour-6h` | 23 | 14.3 | 8.7 | 15 | 2.7 |
| `max`, `hour-6h` (the bar until now) | 16 | 10.8 | **5.2** | 14 | 0.8 |
| `max`, `bins-6h` | 17 | 14.1 | 2.9 | 10 | 1.7 |

The other four arms read 6.1 to 7.4 above chance.

**Beside.** With all 70 events the same arm leads (41.5 above 7.5 by chance); `max`, `hour-6h`
reads 32.2. On the selection months (not blind) `mean`, `bins-72h` leads at 15.8 of 29 above
chance and `max`, `hour-6h` is last of the twelve at 6.1.

**Conclusions.**

- The bar chosen by area, `max`, `hour-6h`, finds about a third as many events above chance as
  the arm chosen, 9.4 events apart, 3.5 standard deviations of that pair's difference under
  the rotations (2.7). It is last or near last above chance on every period read.
- What holds is a family: the mean over the channels with 24 to 72 hours of history. The first
  five arms lie within 2.7 events, inside the noise of about 5, so the arm within the family is
  chosen by noise.
- Neither declared return applies: the chosen arm finds more by the onset (15 against 14) and
  has the largest short events' excess.
- Counted raw, the same rule would choose `mean`, `hour-24h` (23 found, averaged 24.25): the
  same family.
- The prediction held for the family and missed the old bar's distance: it gave a gap of about
  4 events within the noise, the reading 9.4.

## 2026-11-11 — measured: the verdict on the first mission reads the benchmark's F-score at the detector's own alarm beside the events found above chance

**Question.** Which measure should carry the first mission's verdict? The events found at 1 % of
normal windows depend on chance by up to 8 events between arms (above). The satellite benchmark
ranks by the corrected event-wise F0.5, chosen there with the mission's operators: event-wise
recall, and a precision counting the separate runs of false alarms an operator must clear,
scaled by the share of normal time left quiet, at the detector's own alarm (Kotowski et al.,
ESA-ADB, 2024). Reviews of anomaly-detection metrics ask that a measure be checked against
random detections and reported beside its chance level (Sørbø and Ruocco, 2023).

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/telemetry_verdict_measure_report.py` at the commit adding this section |
| Machine | M1 Pro, the CPU; seconds |
| Arms | the no-model departure arms: 12 on the selection months (29 events), 6 on the validation months (61 events); the months 2000 to 2003 left unread for the F-score |
| Candidates | `found` at 1 %; `f_own`, the corrected event-wise F0.5 at each arm's own alarm (set on its fitting side at 2 %, read against each month); `f_budget`, the same at 1 % |
| Criteria, written before the reading | the claim the measure serves; chance's spread over the arms against their own spread (1,000 rotations); agreement of the arms' order between the two periods; pairs of arms a paired bootstrap over months tells apart (95 %, 2,000 draws, months in which an event begins and the rest drawn apart; the alarm at 1 % set once on the whole side) |
| Command | `telemetry_verdict_measure_report.py measure --dir data/report/l38/local/run --out data/report/l47/measure`, then `power` with 10,000 draws |

**The criteria.**

| Measure | Chance's spread / arms' spread (selection; validation) | Order agreement (6 arms) | Pairs apart (selection; validation) |
| --- | --- | --- | --- |
| `found` | 8.0 / 10 (0.80); 7.2 / 10 (0.72) | +0.46 | 17 of 66; 4 of 15 |
| `f_own` | 0.11 / 0.49 (0.23); 0.03 / 0.14 (0.24) | +0.89 | 38 of 66; 11 of 15 |
| `f_budget` | 0.10 / 0.08 (1.29); 0.07 / 0.12 (0.58) | +0.60 | 8 of 66; 7 of 15 |

**What `f_own` is made of.** The arms share one rule for their alarm, yet flag 0.3 to 2.3 % of
the scored normal windows, as drift moves each differently. They raise 14 to 740 runs of false
alarms against 14 to 45 events found, so for most arms precision is 0.04 to 0.25 (0.53 for the
quietest) and the F-score is close to 1.25 times the precision: one more event found moves it by
0.0015 to 0.0043 on the validation months, while halving the false runs about doubles it. On the selection months `max`, `bins-72h`
flags 0.3 % in 14 runs and finds 16 of 29 (0.536); `max`, `hour-6h` flags 1.8 % in 409 runs and
finds 17 (0.048). The rotations keep how alarms cluster, so `f_own`'s chance keeps part of the
precision the measure is meant to read; its value and chance correlate at +0.86 to +0.98.

**Power on the validation months.** The least gain the first test of a Holm family of two
confirms with power 0.8, 3.08 standard deviations of the paired difference over the 15 pairs:

| Verdict | Least | Median | Largest |
| --- | --- | --- | --- |
| `f_own` | 0.025 | 0.108 | 0.175 |
| excess over chance, events of 61 | 5.3 | 9.5 | 11.3 |

**Conclusions.**

- `f_own` serves the product claim, a detector an operator would keep, in the benchmark's own
  terms, and leads on every declared criterion. At these alarms it reads quiet alarms far more
  than events found, so it cannot alone answer whether a detector finds the events.
- The events found above chance answer that question; they are blind to the operator's cost.
- The verdict is therefore two readings, one Holm family: `f_own`, and the events found at 1 %
  less their chance, each against its own no-model bar, with the events found, the false runs
  and the share flagged beside every reading.
- Least gains: 0.11 in `f_own` and 9.5 events of 61 in the excess, the medians over the pairs.
- `f_budget` is weakest and is not carried.

**Limitations.** Six arms on the validation months make the order agreement coarse. The power
uses the spread of real pairs under a normal approximation, not a simulation of a gain. The
criterion on chance reads clustering as chance for the F-scores. The no-model arms are one
family of detectors; detectors over learnt states may cluster their alarms differently.

## 2026-11-11 — declared before the run: the bar for the F-score, chosen on months no arm was read on by it

**Question.** Which no-model arm is the first mission's bar for the F-score verdict? The bar for
the excess is `mean`, `hour-72h` (above).

**Conditions.** As in the declaration of 2026-11-10: the 12 arms on 2000 to 2003, `class_22`
left out, 42 events, 34,914 windows; code at the commit of this declaration. Command:
`telemetry_verdict_measure_report.py choose-f --dir data/report/l47/early --section early --task
esa-mission1-events --leave-out class_22 --out data/report/l47/choice`.

**The rule, stated before the run.**

- *The choice*: the highest `f_own`; then the most found at the own alarm; then the shorter
  span, `hour` before `bins`, `max` before `mean`.
- *Beside*: the events found, the excess at 1 %, the false runs and the share of normal windows
  flagged, for every arm.
- *What follows*: the chosen arm and `mean`, `hour-72h` become the first mission's two bars,
  each for its verdict, in the registered rules with the two least gains; their validation
  answers are measured and the package's detector is checked against the report on each, rank
  by rank within each month, before any campaign reads them.

**Trial.** The command ran whole three times with every month's scores first rotated: the
F-score by chance 0.026 to 0.077, and the arm chosen `max`, `bins-72h` twice (the fewest flags
and false runs) and `mean`, `hour-72h` once (the third fewest false runs). Under chance the
F-score already prefers the arms whose alarms are fewest and longest.

**What has been seen.** A rotation keeps each month's scores, so the trial showed each arm's
share of normal windows flagged on 2000 to 2003 (1.3 to 2.8 %) and, nearly, its runs of alarms
(170 to 646): the false-alarm part of the F-score, which needs no label and can be read on any
unlabelled months. What stays blind is which events the alarms find. On the selection months every arm's
F-score: `max`, `bins-72h` 0.536, `mean`, `bins-72h` 0.280, `mean`, `hour-72h` 0.202, the rest
0.048 to 0.122. On the validation months six arms': 0.060 to 0.197, `max`, `bins-24h` highest.
The events found and the excess on 2000 to 2003, read for the other bar.

**Prediction.** A lean, not a margin: `max`, `bins-72h`, whose alarms are fewest and longest
under the trial and which leads on the selection months; it finds fewer events than `mean`,
`hour-72h` at its own alarm.

**Limitations.** The F-score at these alarms reads mostly the false runs, so the bar is the
quietest detector more than the one finding most. The rule was written after the selection
and validation F-scores were seen; on 2000 to 2003 it is blind in the events found only.

## 2026-11-11 — measured: by the F-score, the mean departure over 72 hours of bins; both verdicts' bars are one family

**Conditions.** As declared, at `a034aa83`: `choose-f` on 2000 to 2003 less `class_22`, 42
events. Output in `data/report/l47/choice/`.

**The choice.** At each arm's own alarm; the excess at 1 % beside.

| Arm | F-score | Found | Excess | False runs | Flagged |
| --- | --- | --- | --- | --- | --- |
| **`mean`, `bins-72h`** (chosen) | **0.113** | 20 | 13.8 | 189 | 0.76 % |
| `max`, `bins-72h` (predicted) | 0.095 | 15 | 7.4 | 171 | 0.69 % |
| `mean`, `hour-72h` (the excess's bar) | 0.082 | 21 | 14.6 | 287 | 1.03 % |
| `mean`, `bins-24h` | 0.074 | 23 | 13.4 | 348 | 1.33 % |
| `mean`, `hour-24h` | 0.068 | 26 | 11.9 | 433 | 1.46 % |
| `max`, `hour-6h` (the bar until now) | 0.042 | 18 | 5.2 | 494 | 1.76 % |

The other six arms read 0.034 to 0.057.

**Conclusions.**

- The two verdicts' bars are two arms of one family, the mean over the channels with 72 hours
  of history: the F-score's bar is second by the excess (13.8 against 14.6), the excess's bar
  third by the F-score. Between them lie 0.03 in the F-score and 0.8 events, both below the
  least gains (0.11; 9.5 events), which were measured on the validation months.
- The bar until now is last but one by the F-score (0.042 against 0.113) and by the excess
  (5.2 against 14.6).
- The choice itself is below the least gain: 0.113 against 0.095, a least gain of 0.11.
- The prediction held for the history and missed the aggregation: over 72 hours of bins the
  mean over the channels finds 20 events where the largest finds 15, at a like number of false
  runs (189 against 171).

**Corrections to the declaration.** The rotated trial did not show the shares flagged on these
months, only near them: 1.3 to 2.8 % under rotation against 0.7 to 2.4 % read, since a rotation
moves high scores off the events onto normal windows. Less was seen than declared; the runs of
alarms were seen nearly as declared (170 to 646 under rotation, 171 to 632 read).

**Limitations found after the run.** Here each arm's own alarm is set on the later months and
read on the earlier, while every verdict sets it on the fitting side and reads later months;
the F-score reads mostly how quiet that alarm stays, which may depend on the direction (the
next section checks it forwards). A window names only the lowest event it overlaps, so an event
wholly under another is not counted: one besides `class_22` here, 3 of 64 on the validation
months, where the benchmark counts each event; the package's harness counts the same way.

**Across the two periods.** Over the 12 arms the F-score orders the arms alike on the selection
months, alarms set forwards, and on 2000 to 2003, alarms set backwards (Spearman +0.83; the
excess +0.70). Both `bins-72h` arms are first and second on both, but the lead flips: 0.536
against 0.280 for `max` on the selection months, 0.113 against 0.095 for `mean` here. With 14
false runs one run more or less moves the F-score of `max`, `bins-72h` by about 0.015 and one
event by 0.02, so the F-score's scale is loose for the quietest detectors, and the least gain of
0.11, measured on louder arms, may not fit them; it is measured again with them below.

## 2026-11-11 — declared before the run: each verdict's bar is a pair of one family, read forwards on the validation months

**Question.** Does each verdict's bar hold on the validation months, with alarms set on the
whole fitting side as every verdict sets them?

**The bars, changed after the readings above.** A single arm chosen within the leading family
is chosen by noise and its lead flips between periods, so each bar is a pair from that family,
read in every reading as the stronger of the two by that reading's measure:

- the F-score: `max`, `bins-72h` and `mean`, `bins-72h`;
- the excess: `mean`, `hour-72h` and `mean`, `bins-72h`.

This change followed the readings. For a candidate it makes a bar harder to pass: it must beat
the stronger of two. For the check below it is easier in one respect: the pair's sibling no
longer challenges the bar.

**Conditions.**

| | |
| --- | --- |
| Code | at the commit of this declaration |
| Answers | the 12 no-model arms on the validation months, 39,384 windows, 61 events: `hour-6h`, `bins-6h`, `bins-24h` from the 2026-11-06 run; `hour-24h`, `hour-72h`, `bins-72h` measured by the same report's `measure --section refit` at this commit, `max` and `mean` each |
| Commands | per arm of `hour-24h`, `hour-72h`, `bins-72h`: `telemetry_channel_local_report.py measure --section refit` with the 2026-11-06 run's task, grid and context, a `chosen.json` naming the arm, the context statistics linked from that run; the twelve arms' answers linked into `data/report/l47/validation` (the untracked `data/report/l47/refit.sh`); then `telemetry_verdict_measure_report.py forward --dir data/report/l47/validation --out data/report/l47/forward --f-bar max:bins-72h mean:bins-72h --excess-bar mean:hour-72h mean:bins-72h --resamples 10000` (1,000 rotations); beside, `telemetry_verdict_measure_report.py choose-f` and `telemetry_channel_local_report.py choose-recall`, each `--dir data/report/l47/validation --section refit --out data/report/l47/forward`, whose arm marked chosen is not used |

**The rule, stated before the run.**

- *Least gains*: for each verdict, 3.08 standard deviations of the paired difference, 10,000
  draws over months, the median over every pair holding an arm of its bar (21 of the 66). These
  are the least gains registered and the ones the check uses; the check itself reads the
  difference to the stronger of the pair, whose spread is not quite a single pair's.
- *The check*: an arm outside a bar beats it when its point gain over the stronger of the pair
  reaches the least gain, or its interval at 97.5 % (the first level of the Holm family) lies
  above nought, the bar's value taken in every draw as the stronger of the pair. If any arm
  beats either bar, nothing is registered: the arms that beat it and the bar go to the user,
  with the readings, for a decision.
- Otherwise the two bars, the two least gains and the rule of two verdicts under Holm are
  registered.
- *Beside*: every arm's F-score, events found, excess, false runs and share flagged.

**Trial.** `refit.sh --trial` measured the six arms in three minutes, 1,704 windows unread as in
the earlier run; nothing was read from them. The same command on `hour-6h` reproduced the
earlier run's validation answers byte for byte, so the old and new answers join. `forward`,
`choose-f` and `choose-recall` ran whole on these answers with every month's scores first
rotated, three times for `forward`: no arm beat a bar at chance, and the `bins-72h` pair read the
highest F-score by chance too.

**What has been seen on these months.** For the six arms of the 2026-11-06 run: the F-score at
the own alarm and at 1 % (0.060 to 0.197, `max`, `bins-24h` highest; 0.115 to 0.238), the events
found and false runs, the excess (15.2 to 32.4, `mean`, `bins-24h` highest), the pairs told
apart and their paired differences; that run also read the area and found share of these arms
and of its taught, state and cell detectors. For `hour-24h`, `hour-72h` and `bins-72h`, either
detector: through the rotated trial, each arm's false runs and share flagged at its own alarm
and its chance at 1 %, near their true values, the `bins-72h` pair the quietest; which events
any of them finds, not at all.

**Prediction.** Neither bar is beaten. Sources: on the selection months, alarms set forwards,
the `bins-72h` pair led the F-score at 0.536 and 0.280, against 0.202 for `mean`, `hour-72h`
and at most 0.122 for the rest, and
`mean`, `bins-72h` led the excess (15.8; `mean`, `hour-72h` third at 12.7); on 2000 to 2003 both
pairs led their measures. Against the arms already seen, the F-score's pair is beaten if its
stronger arm reads below 0.197 less the least gain, or a seen arm's interval clears nought; the
excess's pair likewise against 32.4. These are point readings, with no interval behind them.

**Limitations.** Six of the twelve arms have been read here; the check judges the bars, not the
arms. A bar that is the stronger of two is read with a little upward bias, which makes the
verdicts conservative. The least gains come from no-model arms; detectors over learnt states
may spread otherwise.

## 2026-11-11 — measured: the F-score's pair holds forwards; the excess orders the arms on the validation months unlike before, so its bar is the best arm of each reading

**Conditions.** As declared, at `22e1fd7d`: the six new arms' answers byte for byte those of the
trial; `forward` with 10,000 draws; `choose-f` and `choose-recall` beside. The least gains against
the best of the other arms by `family` at `cffb6114`, 10,000 draws. Output in
`data/report/l47/forward/`.

**The check.** Least gains over the 21 pairs holding an arm of each bar: 0.195 in the F-score,
10.4 events in the excess.

| Verdict | Bar on these months | Strongest other arm | Less the bar [97.5 %] | Arms that beat the bar |
| --- | --- | --- | --- | --- |
| F-score | `max`, `bins-72h` 0.269 (`mean` 0.204) | `max`, `bins-24h` 0.197 | −0.071 [−0.174; −0.021] | none of 10 |
| Excess | `mean`, `hour-72h` 14.6 (`mean`, `bins-72h` 9.4) | `mean`, `bins-24h` 32.4 | +17.8 [+10.6; +26.0] | 7 of 10 |

By the rule nothing was registered and the readings went to a decision.

**The arms' order across the three periods** (Spearman over the 12 arms).

| Measure | 2000–2003 and selection | 2000–2003 and validation | Selection and validation |
| --- | --- | --- | --- |
| F-score | +0.83 | +0.85 | +0.95 |
| Excess | +0.70 | −0.03 | −0.03 |

**Conclusions.**

- The F-score orders the departure arms alike in every period, alarms set backwards or forwards;
  the `bins-72h` pair leads on all three, and its bar holds.
- The excess orders them on the validation months unlike on either earlier period. The arms
  over 72 hours find 13 to 21 of 61 events at 1 % there, against 39 to 42 for the best arms
  over 6 or 24 hours, while at their own alarms, flagging about 2 %, they find 39 to 41: their
  standing depends on the operating point, and the late months differ from the early ones. No
  arm chosen before the validation months is a bar there by the excess.
- The short events gave the warning on 2000 to 2003 too weakly: on the validation months
  `mean`, `bins-72h` finds 5.9 short events above chance against 17.8 for `mean`, `bins-24h`.
- Decision (2026-11-11): the excess's bar is the best of the 12 departure arms in each reading,
  taken in every draw. A candidate must beat every one of them on the same months. The least
  gain against the best of the other eleven, the median over the arms: 10.6 events of 61 (the
  F-score's against the same: 0.195, as for its pair).
- The F-score's bar stays the stronger of `max`, `bins-72h` and `mean`, `bins-72h`.

**Limitations.** The best of twelve read on the same months is biased upward, so the excess
verdict is conservative. Why the order turns is a hypothesis (operating point, drift), not
measured. The package's detector has been checked against this report on `max`, `hour-6h`
only; every arm a campaign reads as a bar is checked first.
