# ADR-0058: Telling the two halves of a window apart — a contrastive term beside masked reconstruction, weighted by the experiment file

- Status: rejected (2026-10-20): the term learnt the stays' measurement clock ([`pretext-variants.md`](../verification/pretext-variants.md))
- Date: 2026-10-19

## Context

Every lever but the objective has been read. Under the mixture of four, masked reconstruction
reads the intensive-care task no better than an untrained encoder from 500 stays and fine-tuning
no better than the network from nothing ([`verdict-statistics.md`](../verification/verdict-statistics.md));
the recipe of adaptation, the domain of the data and the masks changed nothing the reader sees
([`fine-tuning-recipe.md`](../verification/fine-tuning-recipe.md),
[`domain-continuation.md`](../verification/domain-continuation.md),
[`pretext-variants.md`](../verification/pretext-variants.md)), and the objective of states
(ADR-0054) read worse. The encoder's machinery transfers
([`zero-shot-pretext.md`](../verification/zero-shot-pretext.md)); what it learns is the problem:
a hidden value in a dense channel is a line through its neighbours, and a probe reads a window,
not a token. Representations learnt by contrast are more linearly separable than
those learnt by reconstruction (He et al. 2022, Tab. 1), and the one pretext with a gain on these
stays pairs a contrast with a reconstruction (Chowdhury et al. 2023).

## Decision

- **The second view is the complement of the first.** The masks already cut a window in two: the
  tokens the encoder sees and the tokens hidden from it. Each half goes through the encoder as
  the reconstruction's view does, the other half marked as padding. Two disjoint sets of
  observations of one stay or one month must agree on what the window is; no augmentation is
  invented for irregular sets and no second draw is made, so a resumed run takes the same path.
- **The loss is NT-Xent over the micro-batch** (Chen et al. 2020): each half, pooled to one point
  as the task's head pools and carried through two linear maps that are thrown away, must pick
  the other half of its window out of both halves of every other window, by cosine at a stated
  temperature. The windows are of one corpus and never overlap in time (every publication lays
  windows at a stride equal to their length), so a batch holds no false negative, and what
  tells windows apart is what varies within a corpus. The distractors are the term against
  collapse.
- **The term is weighed against one hidden token's loss**: an anchor's mean loss times the weight
  times the hidden tokens scored, so a batch divided by its tokens costs the mean
  reconstruction plus the weight times the mean contrast, and micro-batches weigh as before.
  The trivial predictor is the reconstruction's beside the contrast's chance, `log(2N − 1)`
  per anchor over `N` pairs; the kept epoch is the lowest combined loss over trivial. A window
  the masks hid nothing of is no pair and leaves the contrast, as a distractor too.
- **Two knobs of the experiment file**: `[objective] contrast_weight`, what one anchor's loss
  weighs against one hidden token's (one is the equal weighting of Chowdhury et al.), and
  `contrast_temperature`. Neither is derivable from theory at this batch and loss; the pilot
  before the GPU run reads a small grid of both at tier S and the declaration names the pair.
  A file that states neither renders nothing, so every signature, document and artefact
  written before is unchanged; the artefact carries the term and rebuilds the objective. The
  term is stated beside the value target only.
- **A subclass of masked reconstruction, not a third objective.** The encoder and the decoder are
  built before the projection, so a run under the term draws the same encoder and decoder from
  its seed as the mixture's run and differs from it in the term alone; every diagnostic that
  reads a value objective reads this one. The runtime logs each term's cost per token.

## Consequences

- A step costs about twice the encoder's pass. A micro-batch of 16 gives each anchor 31
  candidates, which is few; a queue of past halves (He et al. 2020) is the remedy if the
  contrast is solved too easily.
- The objective is read by the rule of the stage: pretrained twice on the mixture of four, the
  reader at 500 stays as a screen against harm, and fine-tuning at 200, 50 and 20 stays, where
  the literature's gains live.
- Overlapping windows are one window to the contrast; the test of memorisation on the control
  corpus, whose windows overlap, takes every third.

## Alternatives considered

- **A second independent draw of the masks**: the views then share most tokens and the contrast
  is solved by the shared ones.
- **Crops in time or subsets of channels** (Yue et al. 2022; Zhang et al. 2022): a second
  generator stream to checkpoint, for augmentations the set encoder has no natural form for.
- **Contrast alone**: read below no pretraining at full labels (Chowdhury et al. 2023, Tab. 4).
- **A contrast beside the objective of states**: that objective collapsed its targets.
