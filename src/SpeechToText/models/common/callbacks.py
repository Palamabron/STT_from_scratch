from __future__ import annotations

import lightning.pytorch as pl
from torch.utils.data import DataLoader


class DatasetEpochSync(pl.Callback):
    """Keep dataset augmentations and batch samplers aligned with the trainer epoch."""

    def __init__(self, train_loader: DataLoader) -> None:
        super().__init__()
        self._train_loader = train_loader
        self._ds = train_loader.dataset

    def _sync(self, trainer: pl.Trainer) -> None:
        epoch = int(trainer.current_epoch)
        if hasattr(self._ds, "set_current_epoch"):
            self._ds.set_current_epoch(epoch)
        batch_sampler = getattr(self._train_loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._sync(trainer)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._sync(trainer)
