# Runtime contract

The engine does not choose a named model implementation. It establishes facts
from the container and constructs an operation graph:

- GGUF scalar metadata is retained instead of discarded.
- Tensor names contribute semantic hints; tensor shapes must corroborate them.
- Repeated numeric scopes establish ordering domains without assuming `blk.N`.
- A `verum.graph.json` metadata value can provide an exact declarative graph.
- Unknown matrices are fatal because silently ignoring one can change the model.
- The generated graph is fingerprinted and its MSL is compiled independently on
  every selected Metal device.

The memory invariant for a weight tensor `W[R,C]` is:

`resident(W) <= 2 * windowBytes`, independent of `R*C`.

Each transfer is quant-block aligned. For every row tile and column tile, packed
bytes are copied directly from the mapping into one shared staging window, then
blitted into its paired private window. The MSL linear primitive accumulates
partial column products into the destination row slice. Two window pairs permit
the next tile to be staged without retaining the previous tensor.

Cross-device resources are either remote buffer views inside a nonzero Metal
peer group or explicit destination buffers populated through a shared staging
resource. The destination resource is returned to the scheduler.

## Deliberate refusal boundary

The graph compiler refuses a GGUF when the available metadata, tensor semantics,
and shapes do not prove a unique operation graph. Adding another model-family
case is not the remedy. The remedy is a stronger semantic rule or an embedded
declarative graph that remains expressed in model-independent operations.

