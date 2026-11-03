# ADR-0074: NumPy for the sweep a bootstrap repeats, and nowhere else in the core

- Status: accepted
- Date: 2026-11-03

## Context

The verdict of a campaign is domain code, and its paired bootstrap was stated in plain Python
(ADR-0032) because the sums were per unit and the units few: 180,000 additions. Ranking measures
changed that. An area does not split into a sum per unit, so each resample sweeps every window of
every repeat. A comparison on the first satellite mission is 41,088 windows × 20 repeats × 10,000
resamples, about 8·10⁹ additions: 3.5 ms a sweep and 2.2 hours for the 11 comparisons of one
campaign. The same read holds up every report of pairs, and the clinical tasks pay 45 s a pair.

## Decision

- **Two modules of `evaluation.domain.scoring` import NumPy**: `TiedSweep`, the sweep itself, and
  `UnitCounts`, the draws it reads. Nothing else in the core does. The import contract `pure-core`
  ignores these two imports by name, and a test shows another domain module importing NumPy still
  breaks it. Rankings and pairings take and return plain Python; no array crosses their API.
- **NumPy moves from three extras to the core dependencies**, because any process that reads a
  verdict needs it. All three images already install it, so none grows.
- **One implementation.** The rankings' weighted areas call the sweep; the former loops survive
  only in the tests, as the reference the sweep is held to.
- **The draws do not change.** `random.Random(seed)` still draws every resample in the same order;
  `PairedUnitBootstrap` passes them in batches of 256, which bounds memory and changes no draw.
- **The order of additions holds between groups.** The hits weigh fractions, so the groups are
  accumulated in sweep order; only within a tied group may the last bits differ. The misses are whole numbers, whose sums are exact in any order, so they go through
  a table of misses per unit and edge, by matrix product, or through a running count where that
  table would pass a million entries. An area therefore depends on neither the machine nor its thread count.

## Consequences

- The verdict of a first-mission campaign takes 244 s instead of about 2.5 hours (M1 Pro, 11
  comparisons of 20 repeats, 10,000 resamples), and a comparison of 2,000 clinical stays about 6 s instead of 45 s.
- Re-reading all 51 comparisons and interactions of that mission: 48 tables byte for byte as
  before. Three differ by at most about 10⁻¹³, in rankings with ties, where NumPy groups the
  additions within a large tied group differently; no p-value changed.
- The domain gains its first numeric dependency. It stays free of I/O and of any seed but its own.
- `average_precision_weighted` and `auroc_weighted` keep their signatures, and a single call is
  faster than before on a large ranking and slower on one of a few windows, where it never
  mattered.

## Alternatives considered

- **Plain Python, sums per unit and group computed once.** Zero change to the architecture, but it
  gains 2.2× on the real rankings (one hour a campaign): between two events the normal windows
  still span most units.
- **A port with a NumPy adapter.** Keeps the domain free of NumPy, but the domain calls no port
  (ADR-0010), so the verdict would move out of `CampaignReading` into the application: a large
  refactor with no change in behaviour.
- **NumPy in the reports only.** The verdict stays at 2.2 hours, and two implementations must agree.
- **A matrix product for everything.** Gains about 100×, but fractions summed by a linear algebra
  library round differently across machines and thread counts, and the identity with earlier
  readings becomes a tolerance.

## Revisit when

A sweep over more than about 10⁶ windows makes the batches the bound on memory.
