import json
from typing import Final

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.adapters.documents.fields import Fields
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.run_signature import RunSignature


class PretrainingOrderJson:
    """An order as JSON, the form it is stored in and handed to the machine that trains.

    Compact and key-ordered, so the same order is always the same bytes and placing it twice
    stores one document.
    """

    FORMAT: Final = "emblema.pretraining-order"
    # Bumped when a document of the version before can no longer be read here: a field changed
    # its meaning or a required one was added. A field an older reader ignores costs no bump.
    # Version 2: the configuration states the share of the corpus a run reads.
    # Version 3: it states the reading its hidden tokens are scored by.
    VERSION: Final = 3

    def __init__(self) -> None:
        self._configurations = ExperimentConfigurationDocument()

    def encode(self, order: PretrainingOrder) -> bytes:
        document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "backbone": str(order.backbone),
            "configuration": self._configurations.encode(order.configuration),
            "manifest": Fields.of_ref(order.manifest),
            "run": order.run,
            "git_commit": order.git_commit,
            "signature": order.signature.digest,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> PretrainingOrder:
        """The order those bytes state.

        Raises:
            UnreadableHandoffDocumentError: If the bytes are not an order of a version this
                reads, a field is missing or mistyped, or what is stated is not an order.
        """
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UnreadableHandoffDocumentError(f"order is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise UnreadableHandoffDocumentError("order is not an object")
        if document.get("format") != self.FORMAT or document.get("version") != self.VERSION:
            raise UnreadableHandoffDocumentError(
                f"document is a {document.get('format')!r} of version "
                f"{document.get('version')!r}, not an order this reads"
            )
        fields = Fields(document)
        try:
            return PretrainingOrder(
                backbone=BackboneId.parse(fields.text("backbone")),
                configuration=self._configurations.decode(fields.mapping("configuration")),
                manifest=fields.ref("manifest"),
                run=fields.text("run"),
                git_commit=fields.text("git_commit"),
                signature=RunSignature(fields.text("signature")),
            )
        except ValueError as error:
            raise UnreadableHandoffDocumentError(f"order is not well formed: {error}") from error
