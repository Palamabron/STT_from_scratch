from __future__ import annotations

import json
import multiprocessing as mp
import random
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

from SpeechToText.dataset import (
    DataConfig,
    FeatureConfig,
    FilterConfig,
    ManifestPaths,
    _pcm_to_float32,
    passes_manifest_filters,
)
from SpeechToText.models.common.eval_decoding import (
    MetricKey,
    collect_probs_for_beam,
    compute_wer_cer,
    decode_batch_with_beam,
    decode_batch_with_greedy,
)
from SpeechToText.models.common.inference import (
    ModelType,
    ctc_attention_special_tokens,
    decode_ctc_attention_attention_greedy,
    decode_ctc_attention_joint_beam,
    encode_ctc_attention,
    forward_ctc_log_probs,
    load_lit_module,
    transducer_greedy_decode_batch,
)


@dataclass
class EvaluateConfig:
    """Config for ASR evaluation (CTC, CTC+Attention, or TDT)."""

    checkpoint: str
    tokenizer_model: str
    train_manifest: str
    val_manifest: str
    splits: tuple[str, ...] = ("train", "val")
    kenlm_model: str | None = None
    device: str = "auto"
    sample_rate: int = 16_000
    decode_types: tuple[str, ...] = ("greedy", "beam_kenlm")
    beam_widths: tuple[int, ...] = (32,)
    alphas: tuple[float, ...] = (0.5,)
    betas: tuple[float, ...] = (1.0,)
    joint_alphas: tuple[float, ...] = (0.3, 0.4, 0.5)
    joint_beam_widths: tuple[int, ...] = (10,)
    length_penalties: tuple[float, ...] = (0.0,)
    max_samples_per_split: int | None = None
    audio_key: str = "audio_filepath"
    text_key: str = "text"
    lang_key: str = "language"
    output_csv: str = "results/eval/evaluation_results.csv"
    batch_size: int = 64
    num_workers: int | None = None
    model_type: ModelType = "auto"
    val_max_symbols_per_t: int = 10
    max_duration: float = 16.0
    subsampling_factor: int = 8
    min_speed_factor: float = 0.95
    seed: int = 42
    max_beam_workers: int = 8


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


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return _pcm_to_float32(wav.squeeze(0))


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


def create_pool(
    decode_types: tuple[str, ...], num_workers: int, *, max_workers: int = 8
) -> mp.pool.Pool | None:
    """Worker pool for pyctcdecode batch beam search (requires fork, not spawn)."""
    need_pool = any(d in ("beam", "beam_kenlm") for d in decode_types)
    if not need_pool or num_workers <= 0:
        return None
    if "fork" not in mp.get_all_start_methods():
        logger.warning("fork unavailable on this platform; beam decode runs sequentially")
        return None
    worker_count = min(num_workers, max_workers)
    ctx = mp.get_context("fork")
    logger.info(f"Beam decode pool: fork, workers={worker_count}")
    return ctx.Pool(processes=worker_count)


def _accumulate_decode_metric(
    metrics: dict[MetricKey, dict[str, float]],
    *,
    decode_type: str,
    ref: str,
    hyp: str,
    lang: str,
    beam_width: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    length_penalty: float | None = None,
) -> None:
    wernum, werden, cernum, cerden = compute_wer_cer(ref, hyp)
    if werden == 0 and cerden == 0:
        return
    for lang_key in ("all", lang):
        key: MetricKey = (decode_type, beam_width, alpha, beta, length_penalty, lang_key)
        state = metrics.setdefault(
            key,
            {
                "wer_num": 0.0,
                "wer_den": 0.0,
                "cer_num": 0.0,
                "cer_den": 0.0,
                "count": 0.0,
            },
        )
        state["wer_num"] += float(wernum)
        state["wer_den"] += float(werden)
        state["cer_num"] += float(cernum)
        state["cer_den"] += float(cerden)
        state["count"] += 1.0


def _decode_hybrid_batch(
    *,
    model: torch.nn.Module,
    enc: torch.Tensor,
    out_lengths: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    refs_batch: list[str],
    langs_batch: list[str],
    config: EvaluateConfig,
    tokenizer: SentencePieceProcessor,
    metrics: dict[MetricKey, dict[str, float]],
) -> None:
    tokens = ctc_attention_special_tokens(model)
    batch_size = enc.size(0)

    for index in range(batch_size):
        ref = refs_batch[index]
        lang = langs_batch[index]

        if "attention_greedy" in config.decode_types:
            hyp = decode_ctc_attention_attention_greedy(
                model,
                enc,
                out_lengths,
                index,
                sp=tokenizer,
                tokens=tokens,
            )
            _accumulate_decode_metric(
                metrics,
                decode_type="attention_greedy",
                ref=ref,
                hyp=hyp,
                lang=lang,
            )

        if "joint_beam" not in config.decode_types:
            continue

        for beam_width in config.joint_beam_widths:
            for joint_alpha in config.joint_alphas:
                for length_penalty in config.length_penalties:
                    hyp = decode_ctc_attention_joint_beam(
                        model,
                        enc,
                        out_lengths,
                        ctc_log_probs,
                        index,
                        sp=tokenizer,
                        tokens=tokens,
                        alpha=float(joint_alpha),
                        beam_size=int(beam_width),
                        top_k=int(beam_width),
                        length_penalty=float(length_penalty),
                    )
                    _accumulate_decode_metric(
                        metrics,
                        decode_type="joint_beam",
                        ref=ref,
                        hyp=hyp,
                        lang=lang,
                        beam_width=int(beam_width),
                        alpha=float(joint_alpha),
                        length_penalty=float(length_penalty),
                    )


def process_audio_batch(
    audio_tensors: list[torch.Tensor],
    audio_lengths: list[int],
    refs_batch: list[str],
    langs_batch: list[str],
    config: EvaluateConfig,
    device: torch.device,
    model: torch.nn.Module,
    model_type: ModelType,
    tokenizer: SentencePieceProcessor,
    labels_for_pyctc: list[str],
    decoder_ctc: Any | None,
    decoders_kenlm: dict[tuple[float, float], Any],
    pool: mp.pool.Pool | None,
    metrics: dict[MetricKey, dict[str, float]],
    blank_id: int,
    *,
    decode_types: tuple[str, ...],
) -> None:
    if not audio_tensors:
        return

    padded_audio = pad_sequence(audio_tensors, batch_first=True).to(device)
    lengths_tensor = torch.tensor(audio_lengths, device=device, dtype=torch.long)

    if model_type == "tdt":
        with torch.inference_mode():
            decoded = transducer_greedy_decode_batch(
                model,
                padded_audio,
                lengths_tensor,
                blank_id=blank_id,
                val_max_symbols_per_t=config.val_max_symbols_per_t,
            )
        for index, ids in enumerate(decoded):
            sp_ids = [token_id - 1 for token_id in ids if token_id != blank_id and token_id > 0]
            hyp = "" if not sp_ids else tokenizer.decode_ids(sp_ids)
            ref = refs_batch[index]
            lang = langs_batch[index]
            wernum, werden, cernum, cerden = compute_wer_cer(ref, hyp)
            if werden == 0 and cerden == 0:
                continue
            for lang_key in ("all", lang):
                key = ("greedy", None, None, None, None, lang_key)
                state = metrics.setdefault(
                    key,
                    {
                        "wer_num": 0.0,
                        "wer_den": 0.0,
                        "cer_num": 0.0,
                        "cer_den": 0.0,
                        "count": 0.0,
                    },
                )
                state["wer_num"] += float(wernum)
                state["wer_den"] += float(werden)
                state["cer_num"] += float(cernum)
                state["cer_den"] += float(cerden)
                state["count"] += 1.0
    else:
        hybrid_decode = model_type == "ctc_attention" and any(
            d in decode_types for d in ("attention_greedy", "joint_beam")
        )
        with torch.inference_mode():
            if hybrid_decode:
                enc, batch_out_lengths, batch_log_probs = encode_ctc_attention(
                    model, padded_audio, lengths_tensor
                )
            else:
                batch_log_probs, batch_out_lengths = forward_ctc_log_probs(
                    model, padded_audio, lengths_tensor, model_type
                )
                enc = None

        if "greedy" in decode_types:
            decode_batch_with_greedy(
                batch_log_probs=batch_log_probs,
                batch_lengths=batch_out_lengths,
                batch_refs=refs_batch,
                batch_langs=langs_batch,
                tokenizer=tokenizer,
                blank_id=blank_id,
                metrics=metrics,
            )

        if hybrid_decode and enc is not None:
            _decode_hybrid_batch(
                model=model,
                enc=enc,
                out_lengths=batch_out_lengths,
                ctc_log_probs=batch_log_probs,
                refs_batch=refs_batch,
                langs_batch=langs_batch,
                config=config,
                tokenizer=tokenizer,
                metrics=metrics,
            )

        if any(d in decode_types for d in ("beam", "beam_kenlm")):
            vocab_size_with_blank = len(labels_for_pyctc)
            probs_list, refs_list, langs_list = collect_probs_for_beam(
                batch_log_probs=batch_log_probs,
                batch_lengths=batch_out_lengths,
                batch_refs=refs_batch,
                batch_langs=langs_batch,
                vocab_size_with_blank=vocab_size_with_blank,
            )

            decode_batch_with_beam(
                decode_types=decode_types,
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

    audio_tensors.clear()
    audio_lengths.clear()
    refs_batch.clear()
    langs_batch.clear()


def evaluate_split(
    split_name: str,
    items: list[dict[str, Any]],
    config: EvaluateConfig,
    device: torch.device,
    model: torch.nn.Module,
    model_type: ModelType,
    sample_rate: int,
    tokenizer: SentencePieceProcessor,
    labels_for_pyctc: list[str],
    decoder_ctc: Any | None,
    decoders_kenlm: dict[tuple[float, float], Any],
    *,
    decode_types: tuple[str, ...],
    filter_cfg: FilterConfig,
    feat_cfg: FeatureConfig,
) -> list[dict[str, Any]]:
    if not items:
        return []

    if model_type == "tdt" and "greedy" not in decode_types:
        logger.warning(
            "RNN-T/TDT evaluation supports only greedy decode; add 'greedy' to --decode_types (got {!r})",
            decode_types,
        )
        return []

    blank_id = 0
    metrics: dict[MetricKey, dict[str, float]] = {}

    pool = create_pool(
        decode_types,
        config.num_workers or 0,
        max_workers=config.max_beam_workers,
    )

    try:
        audio_tensors: list[torch.Tensor] = []
        audio_lengths: list[int] = []
        refs_batch: list[str] = []
        langs_batch: list[str] = []

        for example in tqdm(items, desc=f"{split_name} [forward+decode]", leave=False):
            if not passes_manifest_filters(
                example,
                sp=tokenizer,
                filter_cfg=filter_cfg,
                feat_cfg=feat_cfg,
                text_key=config.text_key,
                split=split_name,
            ):
                continue

            audio_path = example.get(config.audio_key) or example.get("audio_path")
            if audio_path is None:
                logger.warning(
                    "Skipping example missing audio path under keys '{}' or 'audio_path'",
                    config.audio_key,
                )
                continue

            text = example.get(config.text_key)
            if not isinstance(text, str) or not text.strip():
                logger.warning("Skipping example with missing or empty text")
                continue

            lang = example.get(config.lang_key, "unknown")

            audio = load_audio(str(audio_path), sample_rate)
            audio_tensors.append(audio)
            audio_lengths.append(int(audio.numel()))
            refs_batch.append(text)
            langs_batch.append(lang)

            if len(audio_tensors) >= config.batch_size:
                process_audio_batch(
                    audio_tensors=audio_tensors,
                    audio_lengths=audio_lengths,
                    refs_batch=refs_batch,
                    langs_batch=langs_batch,
                    config=config,
                    device=device,
                    model=model,
                    model_type=model_type,
                    tokenizer=tokenizer,
                    labels_for_pyctc=labels_for_pyctc,
                    decoder_ctc=decoder_ctc,
                    decoders_kenlm=decoders_kenlm,
                    pool=pool,
                    metrics=metrics,
                    blank_id=blank_id,
                    decode_types=decode_types,
                )

        if audio_tensors:
            process_audio_batch(
                audio_tensors=audio_tensors,
                audio_lengths=audio_lengths,
                refs_batch=refs_batch,
                langs_batch=langs_batch,
                config=config,
                device=device,
                model=model,
                model_type=model_type,
                tokenizer=tokenizer,
                labels_for_pyctc=labels_for_pyctc,
                decoder_ctc=decoder_ctc,
                decoders_kenlm=decoders_kenlm,
                pool=pool,
                metrics=metrics,
                blank_id=blank_id,
                decode_types=decode_types,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    rows: list[dict[str, Any]] = []
    for (decode_type, beam_width, alpha, beta, length_penalty, lang_key), values in metrics.items():
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
                "length_penalty": length_penalty,
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

    group_keys = ["language", "decode_type", "beam_width", "alpha", "beta", "length_penalty"]
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
    sp_vocab = tokenizer.get_piece_size()
    logger.info(f"Loaded SentencePiece tokenizer with vocab_size={sp_vocab}")

    logger.info(f"Loading checkpoint from {config.checkpoint}")
    model, resolved_type = load_lit_module(
        config.checkpoint,
        sp=tokenizer,
        model_type=config.model_type,
    )
    model.eval()
    model.to(device)
    logger.info(f"Loaded model type: {resolved_type}")

    if resolved_type == "ctc_attention":
        attn_tokens = ctc_attention_special_tokens(model)
        logger.info(
            "CTC+Attn token ids: blank={} bos={} eos={} pad={} max_decode_len={}",
            attn_tokens.blank_id,
            attn_tokens.bos_id,
            attn_tokens.eos_id,
            attn_tokens.pad_id,
            attn_tokens.max_decode_len,
        )

    if resolved_type == "tdt" and any(d != "greedy" for d in config.decode_types):
        logger.warning("TDT evaluation supports greedy decoding only; ignoring beam decode types")
    decode_types = (
        ("greedy",)
        if resolved_type == "tdt" and any(d != "greedy" for d in config.decode_types)
        else config.decode_types
    )

    feat_cfg = FeatureConfig(sample_rate=config.sample_rate)
    filter_cfg = FilterConfig(
        max_duration=config.max_duration,
        subsampling_factor=config.subsampling_factor,
        min_speed_factor=config.min_speed_factor,
    )

    _ = DataConfig(
        manifests=ManifestPaths(train=config.train_manifest, val=config.val_manifest),
        tokenizer_model=config.tokenizer_model,
        features=feat_cfg,
        filter=filter_cfg,
    )

    labels_for_pyctc = [""] + [tokenizer.id_to_piece(i) for i in range(sp_vocab)]
    expected_ctc_dim = len(labels_for_pyctc)
    logger.info(f"CTC dim (expected) = {expected_ctc_dim}")

    decoder_ctc, decoders_kenlm = build_decoders(config, labels_for_pyctc)

    allowed_splits = {"train", "val"}
    unknown = set(config.splits) - allowed_splits
    if unknown:
        raise ValueError(f"Unknown splits {sorted(unknown)}; use train and/or val")

    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []
    if "train" in config.splits:
        train_items = load_manifest(config.train_manifest)
    if "val" in config.splits:
        val_items = load_manifest(config.val_manifest)
    logger.info(
        f"Loaded manifests: train={len(train_items)}, val={len(val_items)} "
        f"(splits={','.join(config.splits)})"
    )

    if config.max_samples_per_split is not None:
        rng = random.Random(config.seed)
        if train_items:
            shuffled = train_items.copy()
            rng.shuffle(shuffled)
            train_items = shuffled[: config.max_samples_per_split]
        if val_items:
            shuffled = val_items.copy()
            rng.shuffle(shuffled)
            val_items = shuffled[: config.max_samples_per_split]
        logger.info(
            f"Subsampled to max_samples_per_split={config.max_samples_per_split}: "
            f"train={len(train_items)}, val={len(val_items)}"
        )

    split_items: dict[str, list[dict[str, Any]]] = {}
    if "train" in config.splits:
        split_items["train"] = train_items
    if "val" in config.splits:
        split_items["val"] = val_items

    all_results: list[dict[str, Any]] = []
    for split_name, items in split_items.items():
        if not items:
            continue
        logger.info(f"Evaluating {split_name} with {', '.join(decode_types)} decoding")
        all_results.extend(
            evaluate_split(
                split_name=split_name,
                items=items,
                config=config,
                device=device,
                model=model,
                model_type=resolved_type,
                sample_rate=config.sample_rate,
                tokenizer=tokenizer,
                labels_for_pyctc=labels_for_pyctc,
                decoder_ctc=decoder_ctc,
                decoders_kenlm=decoders_kenlm,
                decode_types=decode_types,
                filter_cfg=filter_cfg,
                feat_cfg=feat_cfg,
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
    main(tyro.cli(EvaluateConfig))
