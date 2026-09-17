import json
from typing import Final

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.adapters.documents.fields import Document, Fields
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome


class PretrainingResultJson:
    """A result as JSON, the form the machine that trained stores it in for this one to read."""

    FORMAT: Final = "emblema.pretraining-result"
    # Bumped when a document of the version before can no longer be read here: a field changed
    # its meaning or a required one was added. A field an older reader ignores costs no bump.
    # Version 2: the configuration states the share of the corpus a run reads.
    # Version 3: it states the reading its hidden tokens are scored by.
    VERSION: Final = 3

    def __init__(self) -> None:
        self._configurations = ExperimentConfigurationDocument()

    def encode(self, result: PretrainingResult) -> bytes:
        corpus = result.corpus
        document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "order": Fields.of_ref(result.order),
            "backbone": str(result.backbone),
            "configuration": self._configurations.encode(result.configuration),
            "corpus": {
                "name": corpus.name,
                "checksum": Fields.of_checksum(corpus.checksum),
                "training_windows": corpus.training_windows,
                "validation_windows": corpus.validation_windows,
                "vocabulary_size": corpus.vocabulary_size,
            },
            "git_commit": result.git_commit,
            "resumed_from": Fields.of_optional_ref(result.resumed_from),
            "outcome": {
                "backbone": Fields.of_ref(result.outcome.backbone),
                "epochs": [self._epoch_document(epoch) for epoch in result.outcome.epochs],
            },
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> PretrainingResult:
        """The result those bytes state.

        Raises:
            UnreadableHandoffDocumentError: If the bytes are not a result of a version this
                reads, a field is missing or mistyped, or what is stated is not a result.
        """
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UnreadableHandoffDocumentError(f"result is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise UnreadableHandoffDocumentError("result is not an object")
        if document.get("format") != self.FORMAT or document.get("version") != self.VERSION:
            raise UnreadableHandoffDocumentError(
                f"document is a {document.get('format')!r} of version "
                f"{document.get('version')!r}, not a result this reads"
            )
        fields = Fields(document)
        try:
            corpus, outcome = fields.fields("corpus"), fields.fields("outcome")
            return PretrainingResult(
                order=fields.ref("order"),
                backbone=BackboneId.parse(fields.text("backbone")),
                configuration=self._configurations.decode(fields.mapping("configuration")),
                corpus=TrainingCorpusShape(
                    name=corpus.text("name"),
                    checksum=corpus.checksum("checksum"),
                    training_windows=corpus.integer("training_windows"),
                    validation_windows=corpus.integer("validation_windows"),
                    vocabulary_size=corpus.integer("vocabulary_size"),
                ),
                git_commit=fields.text("git_commit"),
                resumed_from=fields.optional_ref("resumed_from"),
                outcome=TrainingOutcome(
                    backbone=outcome.ref("backbone"),
                    epochs=tuple(self._epoch(epoch) for epoch in outcome.each("epochs")),
                ),
            )
        except ValueError as error:
            raise UnreadableHandoffDocumentError(f"result is not well formed: {error}") from error

    @staticmethod
    def _epoch_document(epoch: EpochOutcome) -> Document:
        return {
            "epoch": epoch.epoch,
            "training_loss": epoch.training_loss,
            "validation_loss": epoch.validation_loss,
            "hidden_ratio": epoch.hidden_ratio,
            "seconds": epoch.seconds,
            "checkpoint": Fields.of_optional_ref(epoch.checkpoint),
            "backbone": Fields.of_optional_ref(epoch.backbone),
        }

    @staticmethod
    def _epoch(fields: Fields) -> EpochOutcome:
        return EpochOutcome(
            epoch=fields.integer("epoch"),
            training_loss=fields.number("training_loss"),
            validation_loss=fields.number("validation_loss"),
            hidden_ratio=fields.number("hidden_ratio"),
            seconds=fields.number("seconds"),
            checkpoint=fields.optional_ref("checkpoint"),
            backbone=fields.optional_ref("backbone"),
        )
