import pytest

pytest.importorskip("torch")

from emblema.pretraining.domain.masking_strategy import MaskingStrategy

# The mixture the plan asks for: blocks and whole channels carry most of the hiding, single
# tokens are the minority ingredient, and together they take close to half the window.
MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)


@pytest.fixture
def strategy() -> MaskingStrategy:
    return MIXTURE
