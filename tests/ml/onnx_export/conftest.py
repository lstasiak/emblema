import pytest

from tests.ml.onnx_export.exported_encoder import ExportedEncoder, export_small_encoder


@pytest.fixture
def exported() -> ExportedEncoder:
    return export_small_encoder()
