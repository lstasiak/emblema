"""The calling convention of the inference graph a campaign keeps of a neural candidate.

The graph travels through the artifact store as the form named ``onnx`` in a kept candidate's
manifest, and whoever runs it feeds five arrays and reads two outputs by these names. They are
the schema of that message, so they are stated here, in the published language and in nothing
but the standard library: the exporter pairs them with the arguments it traces, and a runtime in
another context builds its feeds from them without loading anything of this context's interior.

The inputs are named as the shared array codec names its fields, in the order the candidate is
called, so a runtime can feed the graph off a batch by attribute; the order is the contract, and
reordering it is a visible change here rather than a silent one in the exporter.
"""

from typing import Final

INPUT_NAMES: Final = ("features", "channel_ids", "timestamps", "timeless", "padding_mask")

# ``embedding`` is not available as a name: it collides with a value the channel embedding
# contributes, and the runtime refuses a graph with a duplicate definition.
POOLED_EMBEDDING: Final = "pooled_embedding"
PREDICTION: Final = "prediction"
OUTPUT_NAMES: Final = (POOLED_EMBEDDING, PREDICTION)
