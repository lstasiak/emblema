from torch import nn

from emblema.evaluation.adapters.torch.lora_linear import LoraLinear
from emblema.evaluation.domain.exceptions import LoraTargetNotFoundError
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec


class LowRankAdaptation:
    """Puts a low-rank update beside every linear layer of a backbone the spec names.

    A target names a run of consecutive segments of a layer's path — ``qkv`` reaches the
    attention's input projection in every block, ``feedforward`` both linears of every
    feed-forward network, ``attention.projection`` the attention's output and not the time
    encoding's — so a spec speaks of the kind of layer it caps rather than of one block. A target
    that reaches no linear layer is refused rather than ignored, before any layer is wrapped: a
    cap written down and silently not applied would report a method it did not run.
    """

    def __init__(self, spec: LoraSpec) -> None:
        self._spec = spec

    def applied_to(self, backbone: nn.Module) -> tuple[str, ...]:
        """Wrap the targeted linear layers in place; return the paths of the layers wrapped.

        Raises:
            LoraTargetNotFoundError: If a target reaches no linear layer of the backbone.
        """
        linears = {
            path: module
            for path, module in backbone.named_modules()
            if isinstance(module, nn.Linear)
        }
        missing = [
            target
            for target in self._spec.targets
            if not any(self._reaches(target, path) for path in linears)
        ]
        if missing:
            raise LoraTargetNotFoundError(
                f"no linear layer of the backbone lies under {missing}; "
                f"its linear layers are {list(linears)}"
            )
        targeted = {path: linear for path, linear in linears.items() if self._targeted(path)}
        for path, linear in targeted.items():
            parent_path, _, attribute = path.rpartition(".")
            parent = backbone.get_submodule(parent_path)
            setattr(parent, attribute, LoraLinear(linear, self._spec))
        return tuple(targeted)

    def _targeted(self, path: str) -> bool:
        return any(self._reaches(target, path) for target in self._spec.targets)

    @staticmethod
    def _reaches(target: str, path: str) -> bool:
        run, segments = target.split("."), path.split(".")
        return any(segments[i : i + len(run)] == run for i in range(len(segments)))
