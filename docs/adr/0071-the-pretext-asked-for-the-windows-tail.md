# ADR-0071: The pretext asked for the window's tail, a forecast scored as the error

- Status: accepted
- Date: 2026-11-01

## Context

A detector scored by the pretext's error (ADR-0068) hides a window's cells under the backbone's
own masks, drawn as in training: whole channels, blocks of a channel's time and single cells,
several draws a window. What it scores is how well a window is reconstructed from the rest of
itself.

Detectors of telemetry in the field score something else: how far the latest readings fall from
what the readings before them foretold. Published pretrained models of time series read anomalies
the same way, by the residual of a forecast. On the first mission's subtle events the
reconstruction read near chance ([composition-screen](../verification/composition-screen.md)),
and whether a forecast reads them otherwise has not been asked. The backbones of the screen were
trained with no tail among their masks, though a block of half a window sometimes reaches its
end; what a forecast from them is worth is a question for the screen.

## Decision

- **Which cells are hidden is a knob of every detector scored by an error**, `hidden`, with two
  values: `own`, the backbone's masks drawn as in training, and `tail`, the last span of every
  channel hidden with everything before it visible. The default stays `own`, so every campaign
  declared so far reads as it was stored; a stored campaign names the choice only where it is
  the tail.
- **A tail is hidden once.** There is nothing to draw, so the recipe asks one draw and refuses
  more; a name that turns both the draws and the tail is refused whichever comes first.
- **The same cells for every answer.** The pretrained backbone, the ridge per channel and the
  line between visible bins fill in the same hidden tail from the same visible cells, so the
  bars move with the detector; with nothing visible after the tail, the line becomes the last
  visible value carried forward.
- **The seam carries the choice**, not the mask. The process that composes both contexts asks
  the stored objective's reading for the tail, and the reading hides the cells whose span is the
  last, as the horizon a training draw would hide; the seeds go unread. Evaluation never sees
  how the cells are laid out.

## Consequences

- The screen of the reformulated telemetry leg reads `pretext_error@hidden=tail` beside the
  backbone's own masks, with the ridge and the interpolation under the same knob and the
  per-channel reading (ADR-0070) available to both.
- A cell under the tail costs one pass through encoder and decoder a window against eight under
  the backbone's own masks.
- The forecast is one span ahead, a sixth of a window; a longer horizon would be another choice,
  and a backbone trained with a tail among its masks another pretraining.
- Whatever the screen chooses was chosen on the two backbones it reads, so it is confirmed on the
  fresh pretrainings of the clinical confirmation, under a rule written before they run.
