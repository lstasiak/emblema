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
