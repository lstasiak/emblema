from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.arrays.exceptions import UnjoinableWindowsError
from emblema.shared.adapters.arrays.joined_windows import JoinedWindows


class JoinedSides:
    """Both sides of a training corpus, each window joined with the windows before it.

    The two sides are joined as one, so a validation window's history reads the training
    windows before it on its axis, as the windows a task scores read the fitting ones.
    """

    @staticmethod
    def of(
        corpus: TrainingCorpus, *, history: int, bins: int
    ) -> tuple[JoinedWindows, JoinedWindows]:
        """The training and validation windows of ``corpus``, each joined with its history.

        Raises:
            InvalidTrainingCorpusError: If the corpus does not say where its windows lie, or its
                windows cannot be joined: none at all, not all one length, overlapping on one
                axis, holding a static feature, or under a history that does not cut into the
                bins.
        """
        if not corpus.placed:
            raise InvalidTrainingCorpusError(
                f"{corpus.name!r} does not say where its windows lie, so none can be joined "
                "with those before it"
            )
        places = [*corpus.training_places, *corpus.validation_places]
        try:
            joined = JoinedWindows.of(
                (corpus.training, corpus.validation),
                axes=[place.axis for place in places],
                starts=[place.start for place in places],
                ends=[place.end for place in places],
                history=history,
                bins=bins,
            )
        except UnjoinableWindowsError as error:
            raise InvalidTrainingCorpusError(f"{corpus.name!r}: {error}") from error
        split = len(corpus.training)
        return joined[:split], joined[split:]
