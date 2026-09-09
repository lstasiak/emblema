import pytest

from tests.ml.onnx_export.attention import AttentionKind
from tests.ml.onnx_export.exported_encoder import ExportedEncoder, export_dummy_encoder


@pytest.fixture(params=list(AttentionKind), ids=[kind.value for kind in AttentionKind])
def exported(request: pytest.FixtureRequest) -> ExportedEncoder:
    kind: AttentionKind = request.param
    return export_dummy_encoder(kind)


@pytest.fixture
def exported_with_sdpa() -> ExportedEncoder:
    return export_dummy_encoder(AttentionKind.SDPA)
