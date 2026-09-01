"""One device synchronisation per step instead of one per logged number.

``float(tensor)`` on a CUDA tensor is a blocking read: the CPU waits for the whole
queued backlog to drain before it can continue. A criterion that reports seven
diagnostics that way pays seven stalls per training step, and on a small student
(MiniLM-22M, batch 32, short sequences) the step is a few milliseconds, so the
stalls are a real fraction of it -- and each one also stops the DataLoader's
prefetch and the next batch's host-to-device copy from overlapping with compute.

:func:`scalar_metrics` stacks the tensors first and reads them once, which is a
single stall regardless of how many numbers the criterion reports. The returned
dict is exactly what building it with ``float()`` would have produced, so nothing
downstream (logging, wandb, the per-step records) can tell the difference.
"""

from __future__ import annotations

import torch


def scalar_metrics(**values: torch.Tensor | float) -> dict[str, float]:
    """Read every named 0-d tensor back to Python in one synchronisation.

    Args:
        **values: metric name -> scalar tensor (any device, any dtype) or a number
            that is already on the host.

    Returns:
        ``{name: float}`` in the order the arguments were given.

    Plain floats are passed through untouched, so a criterion may mix a tensor
    diagnostic with a constant it already knows without special-casing either.
    """
    tensor_names = [name for name, value in values.items() if torch.is_tensor(value)]
    if not tensor_names:
        return {name: float(value) for name, value in values.items()}

    stacked = torch.stack(
        [values[name].detach().reshape(()).float() for name in tensor_names]
    )
    # The one and only device->host read.
    host = stacked.tolist()
    read = dict(zip(tensor_names, host))
    return {
        name: read[name] if name in read else float(value)
        for name, value in values.items()
    }
