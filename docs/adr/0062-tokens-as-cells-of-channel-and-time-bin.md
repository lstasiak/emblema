# ADR-0062: Tokens as cells of channel × time bin, laid out on the host before the same set encoder; a learnt time encoding beside the fixed one

- Status: accepted
- Date: 2026-10-23

## Context

On the intensive-care stays the network trained from nothing loses 0.045 to 0.059 in area to the
same network handed the trees' table at 20, 50 and 200 stays, and nothing from 500 up
(`docs/verification/gap-to-the-trees.md`): with few labels the encoder fails to build what ten
statistics per channel hold. The states of a channel's own tokens carry those statistics; the
mean over the window loses them (`head-and-representation.md`, 2026-10-21). A token is one
reading (ADR-0011), so every level, extreme and count has to be recovered from hundreds of
points through attention. Networks published on irregular clinical series read a window as cells
of one variable over one span of time, marked by whether anything was measured (DuETT;
t-PatchGNN), and a patch model from nothing on a grid led this network by 0.043 [0.018; 0.069]
at 200 stays (`intensive-care-curve.md`, 2026-10-01). ADR-0017 rejected patches along time
because they assume a grid; a cell over a span of time with any number of readings does not.

## Decision

- **A cell is a token.** A window of `bins` equal spans gives, for channel `c` and bin `b`, one
  token with that channel, its time at the bin's centre, and nine features in place of the raw
  value and gap: presence, `log(1 + count)`, the mean, minimum, maximum, first and last of the
  bin's readings, where in the bin the last reading fell, and the bins passed since the channel
  was last observed, as a share of the window. An empty bin carries the channel's last value
  forward in the five value features, zero where there is none, and counts its "since" from the
  window's start when the channel has no earlier reading, a lower bound as the raw gap is. A
  static feature is one timeless cell.
- **Which cells exist is a knob.** Under `cells=window` every channel the window holds gets a
  row of bins. Under `cells=corpus` every channel the corpus names (the manifest's entries of
  that corpus) gets a row, so a channel never measured in a stay stands as empty cells, as the
  trees see its count of zero. On the stays both cost the same; on a corpus whose
  units each report a sixth of its channels the second costs six times the cells. The selection
  reads which the task wants.
- **Laid out on the host, before the encoder.** `BinnedTokens` in `shared/adapters/arrays` turns
  token windows into a `TokenBatch` whose feature axis is nine wide; the five tensors and their
  order (ADR-0007) do not change, and cells stand in the canonical order of tokens. The encoder is
  `SetEncoder` as it is, its value module an injected linear map over nine features, which
  ADR-0017 provided for. The standard parameter count stays the standard composition's; the
  variant's is held by its own test.
- **A learnt time encoding is the second input knob**: the sines and cosines of
  `FourierTimeEncoding` with frequencies and phases as parameters, started at the fixed
  frequencies and zero phase, so a run begins where the fixed encoding stands. This is the
  ablation arm ADR-0017 left behind the same call.
- **Only the network trained from nothing takes either**: pretrained weights fix the value map's
  width and the time module. Cells exclude the grid (two layouts of one window) and the
  window's statistics (they describe the raw readings); a clip bounds a cell's five value
  features as it bounds a raw reading's value.
- **A candidate over cells is kept in its measured form alone.** The inference graph takes raw
  tokens; serving cells needs the host step before the graph that tokenisation already is, and
  is decided when a candidate passes the confirmation on set B.
- **The published corpora are not touched.** Binning in the Catalog would freeze the number of
  bins, a knob the selection turns, into the data and republish every version. Binning inside
  the encoder with scatter operations would give data-dependent shapes, a sum that is not
  deterministic on CUDA and an export with no guarantee.

## Consequences

- Twelve bins over a 48-hour stay give about forty channels × twelve cells, the token count of
  the raw window; the attention cost does not change.
- The pretraining path is set, not built: `EncoderArchitecture` gains an input part and the
  registry a migration when a backbone is pretrained over cells; the loader calls the same
  layout; a pretext over cells can hide cells and predict their presence, as DuETT does.
- Revisit when the loader's throughput over cells, measured in the pilot before pretraining,
  falls short: the cells of a fixed number of bins are then published as a corpus version.

## Alternatives

- **Cells with readings only**: fewer tokens, presence always one; t-PatchGNN's empty patches
  with a presence bit are what beat no patches.
- **The window's statistics as tokens** (ADR-0061): the trees' table handed to the encoder, which
  the diagnostic already measures.
- **mTAN's linear time component**: left out so the learnt encoding starts equal to the fixed one.

## Sources

Labach et al. (2023), DuETT; Zhang et al. (2024), t-PatchGNN; Shukla and Marlin (2021), mTAN;
Lee et al. (2019), Set Transformer; Nie et al. (2023), PatchTST.
