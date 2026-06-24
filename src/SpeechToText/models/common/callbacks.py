from __future__ import annotations

from typing import Any

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


class FileProgressCallback(pl.Callback):
    """Newline progress lines for nohup / log files where Rich/tqdm bars do not refresh."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        accum = max(1, int(trainer.accumulate_grad_batches))
        # Only log after an optimizer step, not on every micro-batch.
        if (batch_idx + 1) % accum != 0:
            return
        log_every = max(1, int(getattr(trainer, "log_every_n_steps", 10)))
        if trainer.global_step % log_every != 0:
            return
        total_batches = int(trainer.num_training_batches or 0)
        if total_batches <= 0:
            return
        frac = (batch_idx + 1) / total_batches
        filled = int(30 * frac)
        bar = "#" * filled + "-" * (30 - filled)
        loss = trainer.callback_metrics.get("train/loss_step")
        loss_s = f"{float(loss):.3f}" if loss is not None else "—"
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"[{bar}] {batch_idx + 1}/{total_batches} "
            f"step={trainer.global_step} loss={loss_s}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        wer = trainer.callback_metrics.get("val/wer/overall")
        cer = trainer.callback_metrics.get("val/cer/overall")
        if wer is None:
            return
        parts = [f"Epoch {trainer.current_epoch + 1} val/wer/overall={float(wer):.3f}"]
        if cer is not None:
            parts.append(f"val/cer/overall={float(cer):.3f}")
        print(" ".join(parts), flush=True)
