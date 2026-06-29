from __future__ import annotations

import torch.nn as nn


def resolve_transducer_modules(net: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Resolve RNN-T/TDT predictor and joint modules from a model network.

    Prefers ``tdt_decoder`` / ``tdt_joint`` when both attention and transducer
    stacks are present on a shared multi-head encoder.
    """
    decoder: nn.Module | None = None
    joint: nn.Module | None = None

    if hasattr(net, "tdt_decoder"):
        decoder = net.tdt_decoder
    elif hasattr(net, "decoder") and not hasattr(net, "dec_proj"):
        decoder = net.decoder

    if hasattr(net, "tdt_joint"):
        joint = net.tdt_joint
    elif hasattr(net, "joint"):
        joint = net.joint

    if decoder is None or joint is None:
        raise AttributeError(
            "Model must have transducer 'decoder'/'joint' or 'tdt_decoder'/'tdt_joint' modules."
        )

    return decoder, joint


def is_transducer_primary_net(net: nn.Module) -> bool:
    """Return True when streaming should use the transducer decoder path."""
    if hasattr(net, "tdt_decoder"):
        return not hasattr(net, "dec_proj")
    return hasattr(net, "joint")
