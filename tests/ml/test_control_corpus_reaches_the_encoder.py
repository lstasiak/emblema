"""The control's pretraining leg: both synthetic layouts travel the real road to the encoder.

Two corpora of different width, sampled differently and sharing no channel, are registered,
frozen, tokenised, archived and read back through the block — the same use cases and adapters a
real corpus goes through, wired by the process the command line assembles. Their windows then
meet in one batch and pass through the encoder that will be pretrained on them.

Nothing here is trained: the self-supervised objective is held to these windows in its own test,
and this one exists so that a failure there is the objective's rather than the road's. The one
claim it makes about the model is the one the road depends on — that a batch mixing two layouts
is a batch the encoder accepts, with no axis for channels to disagree on.
"""

import pytest
import torch

from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_B
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.support.control_corpus import Control
from tests.support.encoders import SMALL

pytestmark = pytest.mark.ml


def test_both_layouts_publish_as_corpora_of_their_own_width(control: Control) -> None:
    first, second = control.manifests
    vocabulary = second.scheme.vocabulary

    assert (first.corpus, second.corpus) == (CONTROL_A.name, CONTROL_B.name)
    assert len(vocabulary.entries_of(first.corpus)) == CONTROL_A.channels + 1
    assert len(vocabulary.entries_of(second.corpus)) == CONTROL_B.channels + 1


def test_one_vocabulary_holds_both_layouts_and_gives_them_no_channel_in_common(
    control: Control,
) -> None:
    first, second = control.manifests
    vocabulary = second.scheme.vocabulary
    ours, theirs = identifiers(first), identifiers(second)

    assert not ours & theirs
    # The second publication extends the first's vocabulary rather than starting one, so every
    # identifier of the first survives in the scheme the second was tokenised under.
    assert ours | theirs == {entry.channel_id for entry in vocabulary.entries}


def test_a_window_of_each_layout_carries_a_timeless_token(control: Control) -> None:
    for which, manifest in enumerate(control.manifests):
        timeless = {
            entry.channel_id
            for entry in manifest.scheme.vocabulary.entries_of(manifest.corpus)
            if entry.timeless
        }
        window = control.windows_of(which)[0]

        assert timeless
        assert timeless & set(window.channel_ids)


def identifiers(manifest: TokenisationManifest) -> set[int]:
    """The channel identifiers this corpus's own channels carry in the shared vocabulary."""
    return {entry.channel_id for entry in manifest.scheme.vocabulary.entries_of(manifest.corpus)}


def test_the_two_layouts_meet_in_one_batch_the_encoder_accepts(control: Control) -> None:
    windows = control.windows_of(0)[:4] + control.windows_of(1)[:4]
    batch = TokenTensors.from_windows(windows)
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(SMALL, control.vocabulary_size).eval()

    states = encoder(*batch.args)

    assert len({len(window) for window in windows}) > 1
    assert states.shape == (len(windows), batch.token_count, SMALL.width)
    assert torch.isfinite(states).all()


def test_the_gradient_of_a_mixed_batch_reaches_both_layouts_channel_embeddings(
    control: Control,
) -> None:
    windows = control.windows_of(0)[:4] + control.windows_of(1)[:4]
    batch = TokenTensors.from_windows(windows)
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(SMALL, control.vocabulary_size)

    encoder(*batch.args).sum().backward()

    # Reached by name rather than by attribute: the encoder holds its channel embedding as a
    # plain module, so the table is only typed where torch looks a parameter up for us.
    gradient = encoder.get_parameter("channel_embedding.table.weight").grad
    assert gradient is not None
    touched = {int(identifier) for identifier in gradient.abs().sum(dim=1).nonzero().flatten()}
    for manifest in control.manifests:
        assert touched & identifiers(manifest)
