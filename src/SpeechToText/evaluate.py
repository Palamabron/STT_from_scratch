from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
import tyro
from loguru import logger
from pyctcdecode import build_ctcdecoder
from sentencepiece import SentencePieceProcessor
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from SpeechToText.dataset import DataConfig
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention
from SpeechToText.utils.audio import build_feature_transforms, extract_features
from SpeechToText.utils.decoding import (
    collect_probs_for_beam,
    decode_batch_with_beam,
    decode_batch_with_greedy,
)


@dataclass
class EvaluateConfig:
    """Config for CTC evaluation."""

    checkpoint: str
    tokenizer_model: str
    train_manifest: str
    val_manifest: str
    kenlm_model: str | None = None
    device: str = "auto"
    sample_rate: int = 16_000
    decode_types: tuple[str, ...] = ("greedy", "beam_kenlm")
    beam_widths: tuple[int, ...] = (32,)
    alphas: tuple[float, ...] = (0.5,)
    betas: tuple[float, ...] = (1.0,)
    max_samples_per_split: int | None = None
    audio_key: str = "audio_filepath"
    text_key: str = "text"
    lang_key: str = "language"
    output_csv: str = "evaluation_results.csv"
    batch_size: int = 64
    num_workers: int | None = None


def load_manifest(path: str) -> list[dict[str, Any]]:
    """Load JSONL manifest as list of dicts."""
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def init_num_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return num_workers
    try:
        return max(1, mp.cpu_count() - 1)
    except NotImplementedError:
        return 1


def build_decoders(
    config: EvaluateConfig,
    labels_for_pyctc: list[str],
) -> tuple[Any | None, dict[tuple[float, float], Any]]:
    decoder_ctc = None
    if "beam" in config.decode_types:
        logger.info("Building CTC beam-search decoder (no LM)")
        decoder_ctc = build_ctcdecoder(
            labels=labels_for_pyctc,
            kenlm_model_path=None,
            alpha=0.0,
            beta=0.0,
        )

    decoders_kenlm: dict[tuple[float, float], Any] = {}
    if "beam_kenlm" in config.decode_types:
        if not config.kenlm_model:
            raise ValueError("decode_types contains 'beam_kenlm' but kenlm_model is not provided")
        lm_path = Path(config.kenlm_model)
        if not lm_path.exists():
            raise FileNotFoundError(f"KenLM model not found: {lm_path}")
        logger.info(f"Building CTC+KenLM decoders from {lm_path}")
        for alpha in config.alphas:
            for beta in config.betas:
                logger.info(f"  -> alpha={alpha}, beta={beta}")
                decoders_kenlm[(alpha, beta)] = build_ctcdecoder(
                    labels=labels_for_pyctc,
                    kenlm_model_path=str(lm_path),
                    alpha=alpha,
                    beta=beta,
                )

    return decoder_ctc, decoders_kenlm


def create_pool(decode_types: tuple[str, ...], num_workers: int) -> mp.pool.Pool | None:
    need_pool = any(d in ("beam", "beam_kenlm") for d in decode_types)
    if not need_pool or num_workers <= 0:
        return None
    ctx = mp.get_context("fork")
    return ctx.Pool(processes=num_workers)


def process_feature_batch(
    feature_tensors: list[torch.Tensor],
    feature_lengths: list[int],
    refs_batch: list[str],
    langs_batch: list[str],
    config: EvaluateConfig,
    device: torch.device,
    model: LitFastConformerCTCAttention,
    tokenizer: SentencePieceProcessor,
    labels_for_pyctc: list[str],
    decoder_ctc: Any | None,
    decoders_kenlm: dict[tuple[float, float], Any],
    pool: mp.pool.Pool | None,
    metrics: dict[tuple[str, int | None, float | None, float | None, str], dict[str, float]],
    blank_id: int,
) -> None:
    if not feature_tensors:
        return

    padded_features = pad_sequence(feature_tensors, batch_first=True).to(device)
    lengths_tensor = torch.tensor(feature_lengths, device=device, dtype=torch.long)

    with torch.inference_mode():
        outputs = model(padded_features, lengths_tensor)
        batch_log_probs = outputs[0]
        batch_out_lengths = outputs[1]

    if "greedy" in config.decode_types:
        decode_batch_with_greedy(
            batch_log_probs=batch_log_probs,
            batch_lengths=batch_out_lengths,
            batch_refs=refs_batch,
            batch_langs=langs_batch,
            tokenizer=tokenizer,
            blank_id=blank_id,
            metrics=metrics,
        )

    vocab_size_with_blank = len(labels_for_pyctc)
    probs_list, refs_list, langs_list = collect_probs_for_beam(
        batch_log_probs=batch_log_probs,
        batch_lengths=batch_out_lengths,
        batch_refs=refs_batch,
        batch_langs=langs_batch,
        vocab_size_with_blank=vocab_size_with_blank,
    )

    decode_batch_with_beam(
        decode_types=config.decode_types,
        beam_widths=config.beam_widths,
        alphas=config.alphas,
        betas=config.betas,
        probs_per_example=probs_list,
        refs=refs_list,
        langs=langs_list,
        decoder_ctc=decoder_ctc,
        decoders_kenlm=decoders_kenlm,
        pool=pool,
        metrics=metrics,
    )

    feature_tensors.clear()
    feature_lengths.clear()
    refs_batch.clear()
    langs_batch.clear()


def evaluate_split(
    split_name: str,
    items: list[dict[str, Any]],
    config: EvaluateConfig,
    device: torch.device,
    model: LitFastConformerCTCAttention,
    data_config: DataConfig,
    mel_spec: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
    tokenizer: SentencePieceProcessor,
    labels_for_pyctc: list[str],
    decoder_ctc: Any | None,
    decoders_kenlm: dict[tuple[float, float], Any],
) -> list[dict[str, Any]]:
    if not items:
        return []

    blank_id = 0
    metrics: dict[
        tuple[str, int | None, float | None, float | None, str],
        dict[str, float],
    ] = {}

    pool = create_pool(config.decode_types, config.num_workers or 0)

    try:
        feature_tensors: list[torch.Tensor] = []
        feature_lengths: list[int] = []
        refs_batch: list[str] = []
        langs_batch: list[str] = []

        for example in tqdm(items, desc=f"{split_name} [forward+decode]", leave=False):
            audio_path = example.get(config.audio_key) or example.get("audio_path")
            if audio_path is None:
                raise KeyError(
                    f"Example is missing audio path under keys "
                    f"'{config.audio_key}' or 'audio_path': {example.keys()}",
                )

            text = example[config.text_key]
            lang = example.get(config.lang_key, "unknown")

            features, feat_len = extract_features(
                audio_path=audio_path,
                data_config=data_config,
                mel_spec=mel_spec,
                amplitude_to_db=amplitude_to_db,
                device=device,
            )

            feature_tensors.append(features)
            feature_lengths.append(feat_len)
            refs_batch.append(text)
            langs_batch.append(lang)

            if len(feature_tensors) >= config.batch_size:
                process_feature_batch(
                    feature_tensors=feature_tensors,
                    feature_lengths=feature_lengths,
                    refs_batch=refs_batch,
                    langs_batch=langs_batch,
                    config=config,
                    device=device,
                    model=model,
                    tokenizer=tokenizer,
                    labels_for_pyctc=labels_for_pyctc,
                    decoder_ctc=decoder_ctc,
                    decoders_kenlm=decoders_kenlm,
                    pool=pool,
                    metrics=metrics,
                    blank_id=blank_id,
                )

        if feature_tensors:
            process_feature_batch(
                feature_tensors=feature_tensors,
                feature_lengths=feature_lengths,
                refs_batch=refs_batch,
                langs_batch=langs_batch,
                config=config,
                device=device,
                model=model,
                tokenizer=tokenizer,
                labels_for_pyctc=labels_for_pyctc,
                decoder_ctc=decoder_ctc,
                decoders_kenlm=decoders_kenlm,
                pool=pool,
                metrics=metrics,
                blank_id=blank_id,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    rows: list[dict[str, Any]] = []
    for (decode_type, beam_width, alpha, beta, lang_key), values in metrics.items():
        if values["count"] == 0:
            continue
        rows.append(
            {
                "split": split_name,
                "language": lang_key,
                "decode_type": decode_type,
                "beam_width": beam_width,
                "alpha": alpha,
                "beta": beta,
                "num_samples": int(values["count"]),
                "wer_num": values["wer_num"],
                "wer_den": values["wer_den"],
                "cer_num": values["cer_num"],
                "cer_den": values["cer_den"],
            },
        )

    return rows


def build_results_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["wer"] = df["wer_num"] / df["wer_den"].clip(lower=1e-8)
    df["cer"] = df["cer_num"] / df["cer_den"].clip(lower=1e-8)

    group_keys = ["language", "decode_type", "beam_width", "alpha", "beta"]
    df_full = df.groupby(group_keys, as_index=False).agg(
        {
            "num_samples": "sum",
            "wer_num": "sum",
            "wer_den": "sum",
            "cer_num": "sum",
            "cer_den": "sum",
        },
    )
    df_full["split"] = "full"
    df_full["wer"] = df_full["wer_num"] / df_full["wer_den"].clip(lower=1e-8)
    df_full["cer"] = df_full["cer_num"] / df_full["cer_den"].clip(lower=1e-8)

    df_all = pd.concat([df, df_full], ignore_index=True)
    df_all = df_all.sort_values(
        by=["split", "language", "decode_type", "wer"],
        ascending=[True, True, True, True],
    )
    return df_all


def main(config: EvaluateConfig) -> None:
    device = get_device(config.device)
    config.num_workers = init_num_workers(config.num_workers)
    logger.info(f"Using device: {device}, num_workers={config.num_workers}")

    tokenizer = SentencePieceProcessor()
    tokenizer.load(config.tokenizer_model)
    vocab_size = tokenizer.get_piece_size()
    logger.info(f"Loaded SentencePiece tokenizer with vocab_size={vocab_size}")

    logger.info(f"Loading checkpoint from {config.checkpoint}")
    model = LitFastConformerCTCAttention.load_from_checkpoint(
        config.checkpoint,
        sp=tokenizer,
        weights_only=False,
    )
    model.eval()
    model.to(device)

    data_config = DataConfig(
        train_manifest=config.train_manifest,
        val_manifest=config.val_manifest,
        tokenizer_model=config.tokenizer_model,
        sample_rate=config.sample_rate,
    )
    mel_spec, amplitude_to_db = build_feature_transforms(data_config)
    mel_spec.to(device)
    amplitude_to_db.to(device)

    labels_for_pyctc = [""] + [tokenizer.id_to_piece(i) for i in range(vocab_size)]
    expected_ctc_dim = len(labels_for_pyctc)
    logger.info(f"CTC dim (expected) = {expected_ctc_dim}")

    decoder_ctc, decoders_kenlm = build_decoders(config, labels_for_pyctc)

    train_items = load_manifest(config.train_manifest)
    val_items = load_manifest(config.val_manifest)
    logger.info(f"Loaded manifests: train={len(train_items)}, val={len(val_items)}")

    if config.max_samples_per_split is not None:
        train_items = train_items[: config.max_samples_per_split]
        val_items = val_items[: config.max_samples_per_split]
        logger.info(
            f"Subsampled to max_samples_per_split={config.max_samples_per_split}: "
            f"train={len(train_items)}, val={len(val_items)}"
        )

    all_results: list[dict[str, Any]] = []
    for split_name, items in {"train": train_items, "val": val_items}.items():
        if not items:
            continue
        logger.info(f"Evaluating {split_name} with {', '.join(config.decode_types)} decoding")
        all_results.extend(
            evaluate_split(
                split_name=split_name,
                items=items,
                config=config,
                device=device,
                model=model,
                data_config=data_config,
                mel_spec=mel_spec,
                amplitude_to_db=amplitude_to_db,
                tokenizer=tokenizer,
                labels_for_pyctc=labels_for_pyctc,
                decoder_ctc=decoder_ctc,
                decoders_kenlm=decoders_kenlm,
            ),
        )

    if not all_results:
        logger.warning("No results collected, nothing to save/log")
        return

    df_all = build_results_dataframe(all_results)
    output_path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False, na_rep="-")
    logger.info(f"Saved evaluation results to: {output_path}")

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logger.info("\n" + df_all.to_string(index=False, na_rep="-"))


if __name__ == "__main__":
    tyro.cli(main)
