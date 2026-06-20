# SpeechToText

End-to-end multilingual speech-to-text (English + Polish) with Fast-Conformer encoders, SentencePiece tokenization, and PyTorch Lightning training.

Supported model heads:

- **CTC** — `SpeechToText.models.ctc` (recommended starting point on a single GPU)
- **CTC + Attention** — `SpeechToText.models.ctc_attention`
- **RNN-T** — `SpeechToText.models.tdt` (default; standard transducer joint)
- **TDT (Token-and-Duration Transducer)** — same module with `--use_tdt true` (duration head + frame-skipping decode)

Token convention: **blank = 0**; SentencePiece token `i` is stored as model id `i + 1`.

---

## Quick start (data + tokenizer already present)

If `data/manifests/final/train_final.jsonl`, `val_final.jsonl`, and `models/spm_unigram_4k_trainval.model` already exist:

```bash
uv sync --extra dev
source .venv/bin/activate

make test                  # optional
make smoke-train           # optional GPU sanity check

make train-ctc-4090        # main baseline

export SPM=models/spm_unigram_4k_trainval.model
uv run python -m SpeechToText.evaluate \
  --checkpoint checkpoints/ctc_4090/last.ckpt \
  --tokenizer_model "$SPM" \
  --train_manifest data/manifests/final/train_final.jsonl \
  --val_manifest data/manifests/final/val_final.jsonl \
  --output_csv results/eval/ctc_4090.csv
```

---

## Full pipeline (from scratch)

Run all commands from the **project root**.

### Step 0 — Install

```bash
uv sync --extra dev
source .venv/bin/activate
```

Create `.env` (or export in shell):

```bash
HF_TOKEN=hf_...          # required for dataset download
WANDB_API_KEY=...        # optional
WANDB_PROJECT=multilingual_asr
```

Disable W&B: add `--use_wandb false` to any train command.

---

### Step 1 — Download audio + build raw manifests

**Skip if** `data/manifests/final/train_final.jsonl` exists and you did not change `configs/data.yaml`.

```bash
make prepare-data
```

- Bucket definitions: [`configs/data.yaml`](configs/data.yaml)
- Requires `HF_TOKEN`. Slow (tens of GB). Writes manifests under `data/manifests/individual/`.

---

### Step 2 — Rebuild capped train/val manifests

**Skip if** finals are already correct.

```bash
make rebuild-manifests
```

| File | Size |
|------|------|
| `data/manifests/final/train_final.jsonl` | ~106k utterances (50k PL / 56k EN) |
| `data/manifests/final/val_final.jsonl` | ~11k utterances |

Backups: `data/manifests/final/backup_<timestamp>/`.

---

### Step 3 — Tokenizer

**Skip if** using shipped `models/spm_unigram_4k_trainval.model`.

```bash
make train-tokenizer       # 4k (default for Makefile presets)

make train-tokenizer-8k    # balanced EN/PL, 8k + coverage report
make tokenizer-coverage    # audit 4k Polish tail chars / unk
```

For 8k, override tokenizer in training:

```bash
export SPM=models/spm_unigram_8k_trainval_balanced.model
# add --data.tokenizer_model "$SPM" to train commands
```

Review `results/tokenizer_coverage_8k.json` after 8k training (Polish digraphs, rare letters).

---

### Step 4 — Augmentation banks (optional)

**Skip if** no local MUSAN/RIR. Training works without them.

```bash
uv run python scripts/build_augment_banks.py \
  --noise-dir /path/to/musan \
  --rir-dir /path/to/rirs
```

Outputs: `data/augment/noise_bank.pt`, `data/augment/rir_bank.pt`.

Disable in training: `--musan_path "" --rirs_path ""`.

---

### Step 5 — Train models

```bash
export TRAIN=data/manifests/final/train_final.jsonl
export VAL=data/manifests/final/val_final.jsonl
export SPM=models/spm_unigram_4k_trainval.model
```

#### Recommended order (RTX 4090)

| Step | Command | Notes |
|------|---------|-------|
| 1 | `make train-ctc-4090` | ~40-50 epochs min; watch `val/wer/overall` |
| 2 | `make init-rnnt-from-ctc` | Copy CTC encoder weights |
| 3 | Train RNN-T with `--ckpt_path` (see below) | Warm-started transducer |
| 4 | `make train-tdt-4090` | After stable RNN-T baseline |
| 5 | `make train-ctc-attn-4090` | Optional |

OOM fallbacks: `train-ctc-4090-oom`, `train-rnnt-4090-oom`, `train-ctc-attn-4090-oom`.

Hyperparameter reference: `configs/train/*.env`.

#### Warm-start RNN-T from CTC

```bash
make init-rnnt-from-ctc

uv run python -m SpeechToText.models.tdt.train \
  --data.manifests.train "$TRAIN" \
  --data.manifests.val "$VAL" \
  --data.tokenizer_model "$SPM" \
  --ckpt_path checkpoints/rnnt_4090/encoder_from_ctc.ckpt \
  --precision bf16-mixed \
  --max_epochs 100 \
  --checkpoint_dir checkpoints/rnnt_4090 \
  --compute_eval_loss false \
  --rnnt_clamp -1.0 \
  --val_max_symbols_per_t 10 \
  --wandb_run_name rnnt-4090-from-ctc
```

Random-init RNN-T (no CTC warm-start): `make train-rnnt-4090`.

#### Resume

```bash
uv run python -m SpeechToText.models.ctc.train \
  --ckpt_path checkpoints/ctc_4090/last.ckpt \
  --max_epochs 100
```

#### Checkpoint averaging (SWA-style)

Average **last N consecutive epoch** checkpoints (not best-WER cherry-picks):

```bash
make average-checkpoints
```

Or in-trainer: `--use_swa true --swa_epoch_start 45 --swa_lrs 1e-4`.

Stratified language batching is on by default (`LoaderConfig.stratify_by_language=true`).

---

### Step 6 — Evaluate

Training logs: `val/wer/overall`, `val/wer/en`, `val/wer/pl`, `val/mer/overall`, worst examples table in W&B.

Offline:

```bash
export SPM=models/spm_unigram_4k_trainval.model

# CTC
uv run python -m SpeechToText.evaluate \
  --checkpoint checkpoints/ctc_4090/last.ckpt \
  --tokenizer_model "$SPM" \
  --train_manifest "$TRAIN" \
  --val_manifest "$VAL" \
  --decode_types greedy beam \
  --output_csv results/eval/ctc_4090.csv

# RNN-T / TDT (greedy only)
uv run python -m SpeechToText.evaluate \
  --checkpoint checkpoints/rnnt_4090/last.ckpt \
  --model_type tdt \
  --tokenizer_model "$SPM" \
  --train_manifest "$TRAIN" \
  --val_manifest "$VAL" \
  --decode_types greedy \
  --val_max_symbols_per_t 10 \
  --output_csv results/eval/rnnt_4090.csv

# CTC + KenLM (hybrid — not directly comparable to E2E RNN-T)
uv run python -m SpeechToText.evaluate \
  --checkpoint checkpoints/ctc_4090/last.ckpt \
  --tokenizer_model "$SPM" \
  --train_manifest "$TRAIN" \
  --val_manifest "$VAL" \
  --kenlm_model lm/pl_5gram.arpa \
  --decode_types greedy beam_kenlm \
  --output_csv results/eval/ctc_4090_kenlm.csv
```

Ablation: `make ablate-kenlm-ctc` (needs trained checkpoints + KenLM file).

---

### Step 7 — Transcribe

```bash
uv run python -m SpeechToText.transcribe \
  --checkpoint checkpoints/ctc_4090/last.ckpt \
  --tokenizer_model "$SPM" \
  path/to/audio.wav
```

RNN-T/TDT: add `--model_type tdt --val_max_symbols_per_t 10`.

---

## Project layout

```
configs/data.yaml          Dataset buckets
configs/train/*.env        RTX 4090 presets
data/manifests/final/      train_final.jsonl, val_final.jsonl
data/augment/              MUSAN/RIR banks
models/                    SentencePiece tokenizers
checkpoints/               Training outputs (gitignored)
results/eval/              Evaluation CSVs
scripts/                   Data prep, tokenizer, checkpoint tools
src/SpeechToText/          Library code
tests/                     Unit tests
```

---

## Makefile reference

| Target | When to use |
|--------|-------------|
| `make prepare-data` | First-time HF download |
| `make rebuild-manifests` | After prepare-data or bucket edits |
| `make train-tokenizer` | Rebuild 4k SPM |
| `make train-tokenizer-8k` | Balanced EN/PL 8k SPM |
| `make tokenizer-coverage` | Polish tail-char / unk audit |
| `make train-ctc-4090` | CTC baseline |
| `make train-rnnt-4090` | RNN-T |
| `make train-ctc-attn-4090` | CTC+Attention |
| `make train-tdt-4090` | True TDT |
| `make init-rnnt-from-ctc` | CTC encoder -> RNN-T |
| `make average-checkpoints` | SWA-style average (last N epochs) |
| `make ablate-subsample-4x` | Subsampling ablation |
| `make ablate-kenlm-ctc` | Hybrid vs E2E ablation |
| `make ablate-rnnt-clamp` | clamp sanity |
| `make smoke-train` | GPU overfit sanity |
| `make test` / `make fmt` / `make types` | CI checks |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM | `*-oom` Makefile target or lower `--data.loader.train_max_batch_duration` |
| RNN-T val OOM | Keep `--compute_eval_loss false` (default) |
| Missing `HF_TOKEN` | Set in `.env` before `make prepare-data` |
| W&B errors | `--use_wandb false` |
| KenLM eval fails | Build `lm/pl_5gram.arpa` separately |
| Tokenizer mismatch | Same `--data.tokenizer_model` for train and eval |
