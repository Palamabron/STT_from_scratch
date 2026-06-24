from __future__ import annotations


def resolve_aux_layer_indices(
    *, n_layers: int, aux_interval: int, aux_layer: int | None
) -> list[int]:
    """Resolve auxiliary CTC tap layers from interval or midpoint config."""
    if aux_interval > 0:
        return [i for i in range(n_layers - 1) if (i + 1) % aux_interval == 0]
    if aux_layer is not None:
        return [aux_layer]
    midpoint = max(0, n_layers // 2 - 1)
    return [midpoint]
