"""The rule that turns a seed into an order, stated once for everything that must replay."""

import hashlib

# Wide enough that two items ranking equal is not a failure mode anyone has to design around.
_DIGEST_SIZE = 16
# Bytes each part's length is written in before the part itself. Eight is more than any string
# anyone will rank by and keeps the framing a fixed width.
_LENGTH_BYTES = 8


def seeded_rank(seed: int, *parts: object) -> bytes:
    """Where one item falls in the order ``seed`` gives, computed from nothing but the item.

    Ranking each item on its own, rather than shuffling a list, makes an order independent of the
    order the items arrived in, of the Python version it was computed on, and — for every item but
    the one or two either side of a cut — of how many other items there are. A corpus that grew
    therefore keeps the order it had, and a run prepared on one machine replays on another.

    The rule lives here because two places that both rank by a digest have to rank by the *same*
    digest: a unit split made by one and an epoch ordered by the other are replayed from the same
    recorded seed, and a change to either must be a change to both.

    Each part is measured before it is read, so no part can be mistaken for two and no two for
    one: a unit named ``"3:17"`` ranks apart from epoch 3 at position 17, whatever separator either
    happens to contain. Joining the parts with a character would have made the two the same item.

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
