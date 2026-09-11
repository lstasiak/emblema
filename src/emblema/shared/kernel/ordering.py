"""The rule that turns a seed into an order, stated once for everything that must replay."""

import hashlib

# Wide enough that two items ranking equal is not a failure mode anyone has to design around.
_DIGEST_SIZE = 16


def seeded_rank(seed: int, *parts: object) -> bytes:
    """Where one item falls in the order ``seed`` gives, computed from nothing but the item.

    Ranking each item on its own, rather than shuffling a list, makes an order independent of the
    order the items arrived in, of the Python version it was computed on, and — for every item but
    the one or two either side of a cut — of how many other items there are. A corpus that grew
    therefore keeps the order it had, and a run prepared on one machine replays on another.

    The rule lives here because two places that both rank by a digest have to rank by the *same*
    digest: a unit split made by one and an epoch ordered by the other are replayed from the same
    recorded seed, and a change to either must be a change to both.

    Args:
        seed: The run's seed.
        parts: What identifies the item, in a fixed order; each is ranked by its ``str``.
    """
    key = ":".join(str(part) for part in (seed, *parts))
    return hashlib.blake2b(key.encode(), digest_size=_DIGEST_SIZE).digest()
