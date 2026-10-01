import pytest

from emblema.serving.application.authorisation.promotion_policy import PromotionPolicy
from emblema.serving.domain.exceptions import OperationNotPermittedError
from emblema.shared.kernel.identity.principal import Principal
from tests.serving.support import OPERATOR, VISITOR

POLICY = PromotionPolicy()


def test_a_caller_granted_nothing_may_neither_promote_nor_withdraw() -> None:
    with pytest.raises(OperationNotPermittedError, match="visitor may not promote"):
        POLICY.permit_promotion(VISITOR)
    with pytest.raises(OperationNotPermittedError, match="visitor may not withdraw"):
        POLICY.permit_withdrawal(VISITOR)


def test_being_allowed_to_promote_does_not_allow_withdrawing() -> None:
    promoter = Principal(subject="promoter", scopes=frozenset({PromotionPolicy.PROMOTE}))

    POLICY.permit_promotion(promoter)
    with pytest.raises(OperationNotPermittedError, match=str(PromotionPolicy.WITHDRAW)):
        POLICY.permit_withdrawal(promoter)


def test_the_operator_may_do_both() -> None:
    POLICY.permit_promotion(OPERATOR)
    POLICY.permit_withdrawal(OPERATOR)
