from enum import StrEnum


class ComputeTier(StrEnum):
    """Hardware class an experiment runs on.

    Scale is a parameter, method is not: reducing scale is legitimate only when the tier is
    declared in the experiment configuration and reported with the result.

    Attributes:
        S: Local machine, MPS or CPU. Smoke runs, ML tests, fine-tuning at label budget <= 200.
        M: Free GPU (Kaggle T4/P100). Pretraining, transfer matrix, ablations; the default tier
            for published results.
        L: Paid GPU (Colab A100 or rented). Only when M is insufficient, with a documented reason.
    """

    S = "S"
    M = "M"
    L = "L"
