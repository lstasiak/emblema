"""Every campaign file in the repository declares a campaign the defining process would accept.

A campaign file is committed before the campaign runs, and the comparison it declares has to
build under the catalogue the process composes: a file that names a variant no arm can be turned
to, a family smaller than its grid or a division its purpose forbids would be refused the day it
is defined, after it was registered. The catalogue is composed as the process composes it, from
the three registers behind one routing, under the settings every campaign of the repository was
defined under: a variant is a knob turned away from those, so under other settings a registered
variant could read as the arm itself.
"""

from pathlib import Path

import pytest

from emblema.config.boosting_settings import BoostingSettings
from emblema.config.convolution_settings import ConvolutionSettings
from emblema.config.lora_settings import LoraSettings
from emblema.config.patch_settings import PatchSettings
from emblema.config.probe_settings import ProbeSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.config.worker_settings import WorkerSettings
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.entrypoints.workers.known_detectors import KnownDetectors
from emblema.entrypoints.workers.known_patch_models import KnownPatchModels
from emblema.evaluation.adapters.campaigns.campaign_file import CampaignFile
from emblema.evaluation.adapters.candidates.routed_candidate_catalogue import (
    RoutedCandidateCatalogue,
)
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from tests.evaluation.support import CAMPAIGN, OPENED_AT, TASK, WEIGHTS

CAMPAIGNS = Path(__file__).resolve().parents[4] / "campaigns"

# The worker settings every campaign in the repository was defined under.
DECLARED = DeclaredWorker(
    WorkerSettings(
        workspace=Path("data/workspace"),
        corpora=Path("data/raw"),
        schedule=ScheduleSettings(
            epochs=30,
            min_steps=2000,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_fraction=0.1,
            final_lr_fraction=0.01,
        ),
        lora=LoraSettings(
            rank=8, alpha=16.0, dropout=0.0, targets="qkv,attention.projection,feedforward"
        ),
        boosting=BoostingSettings(
            rounds=100,
            max_depth=6,
            learning_rate=0.3,
            row_share=1.0,
            feature_share=1.0,
            min_leaf_weight=1.0,
            l2_penalty=1.0,
            threads=4,
        ),
        convolutions=ConvolutionSettings(
            features=9996,
            ridge_penalties=(
                "0.001,0.00464,0.0215,0.1,0.464,2.15,10,46.4,215,1000,"
                "4640,21500,100000,464000,2150000"
            ),
            threads=4,
        ),
        probe=ProbeSettings(ridge_penalties="0.001,0.01,0.1,1,10,100,1000,10000,100000"),
        patch=PatchSettings(
            patch_length=8,
            stride=4,
            width=192,
            heads=3,
            layers=4,
            feedforward_width=768,
            dropout=0.2,
            grid_resolution=1.0,
        ),
    )
)
CATALOGUE = RoutedCandidateCatalogue(
    {
        **dict.fromkeys(
            KnownArms.refs(),
            KnownArms.catalogue(WEIGHTS, DECLARED.lora(), DECLARED.schedule(), DECLARED.probe()),
        ),
        **dict.fromkeys(
            KnownBaselines.refs(),
            KnownBaselines.catalogue(DECLARED.boosting(), DECLARED.convolutions()),
        ),
        **dict.fromkeys(
            KnownPatchModels.refs(),
            KnownPatchModels.catalogue(DECLARED.patch(), DECLARED.schedule()),
        ),
        **dict.fromkeys(KnownDetectors.refs(), KnownDetectors.catalogue(WEIGHTS)),
    }
)


@pytest.mark.parametrize("path", sorted(CAMPAIGNS.glob("*.toml")), ids=lambda path: str(path.stem))
def test_every_campaign_file_in_the_repository_declares_a_campaign(path: Path) -> None:
    declared = CampaignFile.load(path)
    described = {
        ref: CATALOGUE.describe(ref)
        for ref in dict.fromkeys(
            (*declared.competing(), *(choice.variant for choice in declared.tuned_choices()))
        )
    }

    campaign = EvaluationCampaign.designed(
        campaign_id=CAMPAIGN,
        task=TASK,
        purpose=declared.purpose,
        tier=declared.tier,
        design=CampaignDesign(
            candidates=tuple(described[ref] for ref in declared.competing()),
            control=declared.control(),
            endpoint=declared.endpoint(),
            budgets=declared.label_budgets(),
            endpoint_budget=declared.endpoint_budget(),
            seeds=declared.budgets.seeds,
            rules=declared.comparison_rules(),
            bootstrap=declared.paired_bootstrap(),
            inner_holdout=declared.inner_holdout(),
            tuned=declared.tuned_choices(),
            variants=tuple(
                described[ref]
                for ref in dict.fromkeys(choice.variant for choice in declared.tuned_choices())
            ),
            measure=ErrorMeasure.RMSE if declared.measure is None else declared.measure,
        ),
        opened_at=OPENED_AT,
    )

    assert declared.name == path.stem
    assert len(campaign.design.cells()) == (
        len(declared.competing()) * len(declared.label_budgets()) * len(declared.budgets.seeds)
    )
