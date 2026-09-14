"""The rule that turns a seed into an order, stated once for everything that must replay."""

import hashlib

# Wide enough that two items ranking equal is not a failure mode anyone has to design around.
_DIGEST_SIZE = 16
# Bytes each part's length is written in before the part itself. Eight is more than any string
# anyone will rank by and keeps the framing a fixed width.
_LENGTH_BYTES = 8


def seeded_rank(seed: int, *parts: object) -> bytes:
    """Where one item falls in the order ``seed`` gives, computed from nothing but the item.

    Ranking each item alone rather than shuffling a list makes the order independent of arrival
    order, of the Python version and, for all but the items beside a cut, of how many items there
    are: a grown corpus keeps its order and a run replays on another machine. Unit splits and epoch
    orders both rank here, so both replay from one recorded seed. Each part is length-prefixed
    before it is hashed, so a unit named ``"3:17"`` never ranks as epoch 3 at position 17.

    Args:
        seed: The run's seed.
        parts: What identifies the item, in a fixed order; each is ranked by its ``str``.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for part in (seed, *parts):
        encoded = str(part).encode()
        digest.update(len(encoded).to_bytes(_LENGTH_BYTES))
        digest.update(encoded)
    return digest.digest()
