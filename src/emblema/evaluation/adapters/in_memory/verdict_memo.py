from collections import OrderedDict
from threading import Lock

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict


class InMemoryVerdictMemo:
    """Keeps the verdicts a process read last, forgetting the least recently asked past a count.

    The one adapter of its port, and the one a process serves with: a memo shared between
    processes is a verdict stored when its campaign closes, which is another decision. Guarded
    by a lock, because the process that reads verdicts answers requests on several threads.
    """

    def __init__(self, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"a memo keeps at least one verdict, got {capacity}")
        self._capacity = capacity
        self._verdicts: OrderedDict[tuple[CampaignId, int], CampaignVerdict] = OrderedDict()
        self._lock = Lock()

    def recall(self, campaign: CampaignId, revision: int) -> CampaignVerdict | None:
        key = (campaign, revision)
        with self._lock:
            verdict = self._verdicts.get(key)
            if verdict is not None:
                self._verdicts.move_to_end(key)
            return verdict

    def keep(self, campaign: CampaignId, revision: int, verdict: CampaignVerdict) -> None:
        key = (campaign, revision)
        with self._lock:
            self._verdicts[key] = verdict
            self._verdicts.move_to_end(key)
            while len(self._verdicts) > self._capacity:
                self._verdicts.popitem(last=False)
