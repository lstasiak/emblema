from dataclasses import dataclass

from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResult,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrder,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretraining
from emblema.pretraining.application.use_cases.pretrain_backbone import PretrainBackbone


@dataclass(frozen=True)
class Services:
    """The use cases this process can run, each already holding its dependencies.

    Ordering and accepting need the registry, so a process without one — the machine that only
    trains — has neither and says so by their absence rather than by failing when called.
    """

    pretrain_backbone: PretrainBackbone
    fulfil_pretraining_order: FulfilPretrainingOrder
    order_pretraining: OrderPretraining | None
    accept_pretraining_result: AcceptPretrainingResult | None
