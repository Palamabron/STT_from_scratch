from __future__ import annotations

import pytest
import torch.nn as nn

from SpeechToText.models.common.transducer_modules import (
    is_transducer_primary_net,
    resolve_transducer_modules,
)


class _AttentionDecoder(nn.Module):
  pass


class _TdtDecoder(nn.Module):
    def forward(self, tokens: nn.Module) -> nn.Module:
        return tokens


class _Joint(nn.Module):
    pass


class _SharedHybridNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _AttentionDecoder()
        self.dec_proj = nn.Linear(4, 4)
        self.tdt_decoder = _TdtDecoder()
        self.tdt_joint = _Joint()


class _StandaloneTdtNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _TdtDecoder()
        self.joint = _Joint()


def test_resolve_transducer_modules_prefers_tdt_stack() -> None:
    net = _SharedHybridNet()
    decoder, joint = resolve_transducer_modules(net)
    assert decoder is net.tdt_decoder
    assert joint is net.tdt_joint


def test_is_transducer_primary_net_for_shared_hybrid() -> None:
    assert not is_transducer_primary_net(_SharedHybridNet())


def test_is_transducer_primary_net_for_standalone_tdt() -> None:
    assert is_transducer_primary_net(_StandaloneTdtNet())


def test_resolve_transducer_modules_requires_modules() -> None:
    with pytest.raises(AttributeError):
        resolve_transducer_modules(nn.Module())
