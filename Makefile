UV_DEV := uv run --extra dev
UV := uv run

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Bulk HF cache on 16TB HDD (udisks: /dev/sdb2 → /media/kuba/HDD18TB).
HF_CACHE_ROOT ?= /media/kuba/HDD18TB/hf_cache
TRAIN_MANIFEST := data/manifests/final/train_final.jsonl
VAL_MANIFEST := data/manifests/final/val_final.jsonl
SPM_4K := models/spm_unigram_4k_trainval.model
SPM_8K := models/spm_unigram_8k_trainval_balanced.model
SPM_2K := models/spm_unigram_2k_trainval.model
MUSAN_DIR := data/external/musan/noise
RIR_DIR := data/external/rirs/simulated_rirs
KENLM_MODEL := lm/kenlm_en_pl_5gram.arpa
DATA_CONFIG := configs/data.yaml

# Duration-batched training: total seconds of audio per step (see configs/train/*.env)
BATCH_DURATION := 1200
BATCH_DURATION_OOM := 180
BATCH_DURATION_ATTN_OOM := 400
BATCH_DURATION_RNNT_OOM := 800

CTC_V6_CKPT := checkpoints/ctc_4090_65m_v6/last.ckpt
CTC_V6_BEST := checkpoints/ctc_4090_65m_v6/017-val_wer=0.39.ckpt
CTC_V6_BEST_TRAIN := checkpoints/ctc_4090_65m_v6/025-val_wer=0.36.ckpt
TDT_65M_INIT := checkpoints/tdt_4090_65m/encoder_from_ctc_65m.ckpt

TRAIN_PATHS := \
	--data.manifests.train $(TRAIN_MANIFEST) \
	--data.manifests.val $(VAL_MANIFEST) \
	--data.tokenizer_model $(SPM_4K)

TRAIN_PATHS_2K := \
	--data.manifests.train $(TRAIN_MANIFEST) \
	--data.manifests.val $(VAL_MANIFEST) \
	--data.tokenizer_model $(SPM_2K)

.PHONY: fmt prepare-data prepare-data-600h prepare-data-800h prefetch-hf-600h prefetch-hf-800h rebuild-manifests rebuild-manifests-600h rebuild-manifests-800h analyze-manifests \
	train-tokenizer train-tokenizer-2k train-tokenizer-8k preview-augmentations \
	tokenizer-coverage tokenizer-coverage-2k download-augment-data build-augment-banks \
	test smoke-train types \
	train-ctc-4090 train-rnnt-4090 train-ctc-attn-4090 train-tdt-4090 \
	train-ctc-4090-oom train-ctc-4090-sm train-ctc-4090-65m train-ctc-4090-65m-v2 train-ctc-4090-65m-v3 train-ctc-4090-65m-v4 train-ctc-4090-65m-v5 train-ctc-4090-65m-v6 train-ctc-4090-65m-v6-resume train-ctc-4090-65m-v7 \
	train-rnnt-4090-oom train-ctc-attn-4090-oom init-tdt-from-ctc-65m train-tdt-4090-65m \
	init-rnnt-from-ctc init-rnnt-from-ctc-v2 average-checkpoints \
	ablate-subsample-4x ablate-kenlm-ctc ablate-rnnt-clamp eval-ctc-4090-65m-v5

fmt:
	cd $(REPO_ROOT) && $(UV_DEV) ruff format src scripts tests
	cd $(REPO_ROOT) && $(UV_DEV) ruff check src scripts tests --fix

prepare-data:
	cd $(REPO_ROOT) && PYTHONPATH=scripts $(UV) python -m prepare_data --config $(DATA_CONFIG)

prepare-data-600h:
	@test -d $(HF_CACHE_ROOT) || (echo "Mount HDD: udisksctl mount -b /dev/sdb2" && exit 1)
	@mkdir -p $(HF_CACHE_ROOT)/hub $(HF_CACHE_ROOT)/datasets
	@echo "HF cache: $(HF_CACHE_ROOT) (hub + datasets on HDD18TB)"
	cd $(REPO_ROOT) && PREPARE_DATA_NUM_WORKERS=16 PREPARE_DATA_FETCH_SHARDS=2 \
		HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120 \
		HF_HUB_CACHE=$(HF_CACHE_ROOT)/hub \
		HF_DATASETS_CACHE=$(HF_CACHE_ROOT)/datasets \
		$(MAKE) prepare-data DATA_CONFIG=configs/data_600h.yaml

prepare-data-800h:
	@test -d $(HF_CACHE_ROOT) || (echo "Mount HDD: udisksctl mount -b /dev/sdb2" && exit 1)
	@mkdir -p $(HF_CACHE_ROOT)/hub $(HF_CACHE_ROOT)/datasets
	@echo "HF cache: $(HF_CACHE_ROOT) (hub + datasets on HDD18TB)"
	cd $(REPO_ROOT) && PREPARE_DATA_NUM_WORKERS=16 PREPARE_DATA_FETCH_SHARDS=2 \
		HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120 \
		HF_HUB_CACHE=$(HF_CACHE_ROOT)/hub \
		HF_DATASETS_CACHE=$(HF_CACHE_ROOT)/datasets \
		$(MAKE) prepare-data DATA_CONFIG=configs/data_800h.yaml

# Prefetch CV21 audio tarballs into HF cache (avoids SSL timeouts during streaming).
prefetch-hf-600h:
	@test -f $(REPO_ROOT)/.env || (echo "Missing .env with HF_TOKEN" && exit 1)
	@test -d $(HF_CACHE_ROOT) || (echo "Mount HDD: udisksctl mount -b /dev/sdb2" && exit 1)
	@mkdir -p $(HF_CACHE_ROOT)/hub
	cd $(REPO_ROOT) && set -a && . ./.env && set +a && \
		HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120 \
		HF_HUB_CACHE=$(HF_CACHE_ROOT)/hub \
		$(UV) hf download fsicoli/common_voice_21_0 \
			--repo-type dataset \
			--include "audio/pl/train/*.tar" "audio/en/train/*.tar" \
			--max-workers 4

mount-hdd-cache:
	udisksctl mount -b /dev/sdb2
	@mkdir -p $(HF_CACHE_ROOT)/hub $(HF_CACHE_ROOT)/datasets

# One-time: copy existing hub blobs from home SSD to HDD (skip if hub already on HDD).
migrate-hf-hub-to-hdd:
	@test -d $(HF_CACHE_ROOT) || (echo "Run: make mount-hdd-cache" && exit 1)
	@mkdir -p $(HF_CACHE_ROOT)/hub
	@if [ -d "$(HOME)/.cache/huggingface/hub/datasets--fsicoli--common_voice_21_0" ] && [ ! -d "$(HF_CACHE_ROOT)/hub/datasets--fsicoli--common_voice_21_0" ]; then \
		echo "Copying HF hub (~72G CV21 tars) to $(HF_CACHE_ROOT)/hub ..."; \
		rsync -a --info=progress2 "$(HOME)/.cache/huggingface/hub/" "$(HF_CACHE_ROOT)/hub/"; \
	else \
		echo "Hub already on HDD or home hub missing, skipping."; \
	fi

# Drop HF datasets Arrow cache for exhausted non-streaming buckets (keeps CV21 EN cache).
clean-hf-cache-exhausted:
	rm -rf $(REPO_ROOT)/data/.hf_datasets_cache/amu-cai___pl-asr-bigos-v2
	rm -rf $(REPO_ROOT)/data/.hf_datasets_cache/fsicoli___common_voice_21_0/pl

# Free /mnt/praca after failed non-streaming prepare (partial tar extract + downloads).
clean-hf-cache-mnt-praca:
	rm -rf $(REPO_ROOT)/data/.hf_datasets_cache

# Free home SSD after migrate-hf-hub-to-hdd (only if HDD copy verified).
clean-hf-hub-home:
	@test -d "$(HF_CACHE_ROOT)/hub/datasets--fsicoli--common_voice_21_0" || (echo "Migrate to HDD first" && exit 1)
	rm -rf "$(HOME)/.cache/huggingface/hub"

prefetch-hf-800h: prefetch-hf-600h

rebuild-manifests:
	cd $(REPO_ROOT) && $(UV) python scripts/rebuild_final_manifests.py --config $(DATA_CONFIG)

rebuild-manifests-600h:
	$(MAKE) rebuild-manifests DATA_CONFIG=configs/data_600h.yaml

rebuild-manifests-800h:
	$(MAKE) rebuild-manifests DATA_CONFIG=configs/data_800h.yaml

analyze-manifests:
	cd $(REPO_ROOT) && $(UV) python scripts/manifest_durations_analysis.py \
		--train-manifest $(TRAIN_MANIFEST) \
		--val-manifest $(VAL_MANIFEST)

train-tokenizer:
	cd $(REPO_ROOT) && $(UV) python scripts/train_tokenizer.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--corpus-out data/corpus/sp_trainval.txt \
		--model-prefix models/spm_unigram_4k_trainval \
		--vocab-size 4096 \
		--model-type unigram

train-tokenizer-2k:
	cd $(REPO_ROOT) && $(UV) python scripts/train_tokenizer.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--corpus-out data/corpus/sp_trainval_balanced_2k.txt \
		--model-prefix models/spm_unigram_2k_trainval \
		--vocab-size 2048 \
		--model-type unigram \
		--balance-languages
	cd $(REPO_ROOT) && $(UV) python scripts/tokenizer_coverage_report.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--model-path $(SPM_2K) \
		--output-json results/tokenizer_coverage_2k.json

train-tokenizer-8k:
	cd $(REPO_ROOT) && $(UV) python scripts/train_tokenizer.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--corpus-out data/corpus/sp_trainval_balanced_8k.txt \
		--model-prefix models/spm_unigram_8k_trainval_balanced \
		--vocab-size 8192 \
		--model-type unigram \
		--balance-languages
	cd $(REPO_ROOT) && $(UV) python scripts/tokenizer_coverage_report.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--model-path $(SPM_8K) \
		--output-json results/tokenizer_coverage_8k.json

tokenizer-coverage:
	cd $(REPO_ROOT) && $(UV) python scripts/tokenizer_coverage_report.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--model-path $(SPM_4K) \
		--output-json results/tokenizer_coverage_4k.json

tokenizer-coverage-2k:
	cd $(REPO_ROOT) && $(UV) python scripts/tokenizer_coverage_report.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--model-path $(SPM_2K) \
		--output-json results/tokenizer_coverage_2k.json

download-augment-data:
	cd $(REPO_ROOT) && $(UV) python scripts/download_augment_data.py --dest data/external

build-augment-banks:
	@test -d $(MUSAN_DIR) || (echo "Missing $(MUSAN_DIR). Run: make download-augment-data" && exit 1)
	@test -d $(RIR_DIR) || (echo "Missing $(RIR_DIR). Run: make download-augment-data" && exit 1)
	cd $(REPO_ROOT) && $(UV) python scripts/build_augment_banks.py \
		--noise-dir $(MUSAN_DIR) \
		--rir-dir $(RIR_DIR) \
		--out-noise-bank data/augment/noise_bank.pt \
		--out-rir-bank data/augment/rir_bank.pt
	@test -s data/augment/noise_bank.pt || (echo "noise_bank.pt is empty" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "rir_bank.pt is empty" && exit 1)

preview-augmentations:
	cd $(REPO_ROOT) && $(UV) python scripts/preview_augmentations.py

test:
	cd $(REPO_ROOT) && $(UV_DEV) pytest tests/

smoke-train:
	cd $(REPO_ROOT) && \
	$(UV) python -m SpeechToText.models.ctc.train \
		--data.manifests.train data/debug/en_one.jsonl \
		--data.manifests.val data/debug/en_one.jsonl \
		--data.tokenizer_model $(SPM_4K) \
		--data.loader.train_batch_size 1 \
		--data.loader.val_batch_size 1 \
		--data.filter.max_duration 25.0 \
		--max_epochs 180 \
		--model.encoder.d_model 128 \
		--model.encoder.n_layers 2 \
		--model.encoder.n_heads 2 \
		--optimizer.lr 3e-4 \
		--precision 32-true \
		--musan_path "" \
		--rirs_path "" \
		--no-use-wandb \
		--wandb_run_name debug-overfit-one

types:
	cd $(REPO_ROOT) && $(UV_DEV) mypy src

# --- RTX 4090 presets (see configs/train/*.env for reference values) ---

train-ctc-4090:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090 \
		--model.encoder.d_model 512 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.aux_layer 7 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--ctc_label_smoothing 0.1 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 2e-3 \
		--optimizer.warmup_ratio 0.1 \
		--wandb_run_name ctc-4090

train-ctc-4090-oom:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090 \
		--model.encoder.d_model 512 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.aux_layer 7 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION_OOM) \
		--data.loader.train_max_batch_size 48 \
		--ctc_label_smoothing 0.1 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 2e-3 \
		--wandb_run_name ctc-4090-oom

train-ctc-4090-sm:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_sm \
		--model.encoder.d_model 384 \
		--model.encoder.n_layers 14 \
		--model.encoder.n_heads 6 \
		--model.aux_layer 6 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--ctc_label_smoothing 0.1 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 2e-3 \
		--optimizer.warmup_ratio 0.1 \
		--wandb_run_name ctc-4090-sm

train-ctc-4090-65m:
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--ctc_label_smoothing 0.1 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 2e-3 \
		--optimizer.warmup_ratio 0.1 \
		--wandb_run_name ctc-4090-65m

train-ctc-4090-65m-v2:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v2 \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 0 \
		--audio_augment_start_epoch 0 \
		--audio_augment.heavy_augment_start_epoch 0 \
		--optimizer.lr 2e-3 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--wandb_run_name ctc-4090-65m-v2

train-ctc-4090-65m-v3:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v3 \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 10 \
		--audio_augment_start_epoch 10 \
		--audio_augment.heavy_augment_start_epoch 20 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--wandb_run_name ctc-4090-65m-v3

train-ctc-4090-65m-v4:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f checkpoints/ctc_4090_65m_v3/last.ckpt || (echo "Missing v3 checkpoint. Train or copy last.ckpt first." && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 50 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v4 \
		--ckpt_path checkpoints/ctc_4090_65m_v3/last.ckpt \
		--reset_optimizer_state \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment.time_masks 2 \
		--spec_augment.time_width_fraction 0.05 \
		--spec_augment_start_epoch 0 \
		--audio_augment_start_epoch 0 \
		--audio_augment.heavy_augment_start_epoch 0 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 5e-4 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name ctc-4090-65m-v4

train-ctc-4090-65m-v5:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f checkpoints/ctc_4090_65m_v4/007-val_wer=0.32.ckpt || (echo "Missing v4 best checkpoint." && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 40 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v5 \
		--ckpt_path checkpoints/ctc_4090_65m_v4/007-val_wer=0.32.ckpt \
		--reset_optimizer_state \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment.time_masks 2 \
		--spec_augment.time_width_fraction 0.05 \
		--spec_augment_start_epoch 0 \
		--audio_augment_start_epoch 0 \
		--audio_augment.heavy_augment_start_epoch 0 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 5e-4 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name ctc-4090-65m-v5

eval-ctc-4090-65m-v5:
	@test -f $(KENLM_MODEL) || (echo "Missing $(KENLM_MODEL)" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K)" && exit 1)
	@CKPT="checkpoints/ctc_4090_65m_v5/last.ckpt"; \
	if [ ! -f "$$CKPT" ]; then CKPT="checkpoints/ctc_4090_65m_v4/007-val_wer=0.32.ckpt"; fi; \
	echo "Evaluating $$CKPT on val only"; \
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.evaluate \
		--checkpoint "$$CKPT" \
		--tokenizer_model $(SPM_2K) \
		--train_manifest data/debug/en_one.jsonl \
		--val_manifest $(VAL_MANIFEST) \
		--decode_types greedy beam_kenlm \
		--kenlm_model $(KENLM_MODEL) \
		--batch_size 8 \
		--output_csv results/eval/ctc_4090_65m_v5_val.csv

train-ctc-4090-65m-v6:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f $(TRAIN_MANIFEST) || (echo "Missing $(TRAIN_MANIFEST). Run: make prepare-data-800h && make rebuild-manifests-800h" && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v6 \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment.time_masks 6 \
		--spec_augment.time_width_fraction 0.10 \
		--spec_augment_start_epoch 10 \
		--audio_augment_start_epoch 10 \
		--audio_augment.heavy_augment_start_epoch 20 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name ctc-4090-65m-v6

train-ctc-4090-65m-v7:
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f $(TRAIN_MANIFEST) || (echo "Missing $(TRAIN_MANIFEST). Run: make prepare-data-800h && make rebuild-manifests-800h" && exit 1)
	@test -f $(CTC_V6_BEST) || (echo "Missing $(CTC_V6_BEST). Train v6 first." && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v7 \
		--ckpt_path $(CTC_V6_BEST) \
		--reset_optimizer_state \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--data.loader.num_workers 4 \
		--data.loader.no-persistent-workers \
		--data.loader.prefetch-factor 2 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment.time_masks 6 \
		--spec_augment.time_width_fraction 0.10 \
		--spec_augment_start_epoch 10 \
		--audio_augment_start_epoch 10 \
		--audio_augment.heavy_augment_start_epoch 20 \
		--audio_augment.clean_pass_prob 0.08 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.03 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name ctc-4090-65m-v7

train-ctc-4090-65m-v6-resume:
	@test -f $(CTC_V6_CKPT) || (echo "Missing $(CTC_V6_CKPT). Run: make train-ctc-4090-65m-v6" && exit 1)
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f $(TRAIN_MANIFEST) || (echo "Missing $(TRAIN_MANIFEST). Run: make prepare-data-800h && make rebuild-manifests-800h" && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/ctc_4090_65m_v6 \
		--ckpt_path $(CTC_V6_CKPT) \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--model.aux_layer 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--data.loader.train_max_batch_size 64 \
		--data.loader.num_workers 4 \
		--data.loader.no-persistent-workers \
		--data.loader.prefetch-factor 2 \
		--accumulate_grad_batches 2 \
		--ctc_label_smoothing 0.0 \
		--aux_ctc_weight 0.3 \
		--spec_augment.time_masks 6 \
		--spec_augment.time_width_fraction 0.10 \
		--spec_augment_start_epoch 10 \
		--audio_augment_start_epoch 10 \
		--audio_augment.heavy_augment_start_epoch 20 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name ctc-4090-65m-v6

init-tdt-from-ctc-65m:
	@test -f $(CTC_V6_BEST_TRAIN) || (echo "Missing $(CTC_V6_BEST_TRAIN). Train CTC v6 first." && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	cd $(REPO_ROOT) && $(UV) python scripts/init_encoder_from_checkpoint.py \
		--source-checkpoint $(CTC_V6_BEST_TRAIN) \
		--tokenizer-model $(SPM_2K) \
		--target rnnt \
		--output $(TDT_65M_INIT)

train-tdt-4090-65m: init-tdt-from-ctc-65m
	@test -s data/augment/noise_bank.pt || (echo "Missing data/augment/noise_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -s data/augment/rir_bank.pt || (echo "Missing data/augment/rir_bank.pt. Run: make build-augment-banks" && exit 1)
	@test -f $(SPM_2K) || (echo "Missing $(SPM_2K). Run: make train-tokenizer-2k" && exit 1)
	@test -f $(TRAIN_MANIFEST) || (echo "Missing $(TRAIN_MANIFEST). Run: make prepare-data-800h && make rebuild-manifests-800h" && exit 1)
	@test -f $(TDT_65M_INIT) || (echo "Missing $(TDT_65M_INIT). Run: make init-tdt-from-ctc-65m" && exit 1)
	cd $(REPO_ROOT) && PYTHONUNBUFFERED=1 $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS_2K) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/tdt_4090_65m \
		--ckpt_path $(TDT_65M_INIT) \
		--use-tdt \
		--tdt_sigma 0.05 \
		--tdt_omega 0.1 \
		--model.encoder.d_model 400 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--model.encoder.conv_kernel 9 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION_RNNT_OOM) \
		--data.loader.train_max_batch_size 48 \
		--data.loader.num_workers 4 \
		--data.loader.no-persistent-workers \
		--data.loader.prefetch-factor 2 \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--val_max_symbols_per_t 10 \
		--joint_fused_batch_size 2 \
		--spec_augment.time_masks 6 \
		--spec_augment.time_width_fraction 0.10 \
		--spec_augment_start_epoch 10 \
		--audio_augment_start_epoch 10 \
		--audio_augment.heavy_augment_start_epoch 20 \
		--audio_augment.clean_pass_prob 0.08 \
		--audio_augment.bg_noise_prob 0.25 \
		--audio_augment.rir_prob 0.2 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.05 \
		--optimizer.scheduler cosine \
		--optimizer.cosine_eta_min 1e-5 \
		--wandb_run_name tdt-4090-65m

train-rnnt-4090:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/rnnt_4090 \
		--model.encoder.d_model 384 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--val_max_symbols_per_t 10 \
		--joint_fused_batch_size 4 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 1e-3 \
		--wandb_run_name rnnt-4090

train-rnnt-4090-oom:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/rnnt_4090 \
		--model.encoder.d_model 384 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION_RNNT_OOM) \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--val_max_symbols_per_t 10 \
		--joint_fused_batch_size 2 \
		--wandb_run_name rnnt-4090-oom

train-ctc-attn-4090:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc_attention.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 120 \
		--checkpoint_dir checkpoints/ctc_attn_4090 \
		--model.encoder.d_model 256 \
		--model.encoder.n_layers 12 \
		--model.encoder.n_heads 4 \
		--model.aux_layer 5 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--ctc_weight 0.3 \
		--aux_ctc_weight 0.3 \
		--ctc_label_smoothing 0.1 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 10 \
		--optimizer.lr 1e-3 \
		--optimizer.warmup_ratio 0.15 \
		--wandb_run_name ctc-attn-4090

train-ctc-attn-4090-oom:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc_attention.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 120 \
		--checkpoint_dir checkpoints/ctc_attn_4090 \
		--model.encoder.d_model 256 \
		--model.encoder.n_layers 12 \
		--model.encoder.n_heads 4 \
		--model.aux_layer 5 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION_ATTN_OOM) \
		--ctc_weight 0.3 \
		--wandb_run_name ctc-attn-4090-oom

train-tdt-4090:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/tdt_4090 \
		--use-tdt \
		--tdt_sigma 0.05 \
		--tdt_omega 0.1 \
		--model.encoder.d_model 384 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--val_max_symbols_per_t 10 \
		--joint_fused_batch_size 4 \
		--wandb_run_name tdt-4090

init-rnnt-from-ctc:
	cd $(REPO_ROOT) && $(UV) python scripts/init_encoder_from_checkpoint.py \
		--source-checkpoint checkpoints/ctc_4090/last.ckpt \
		--tokenizer-model $(SPM_4K) \
		--target rnnt \
		--output checkpoints/rnnt_4090/encoder_from_ctc.ckpt

init-rnnt-from-ctc-v2:
	cd $(REPO_ROOT) && $(UV) python scripts/init_encoder_from_checkpoint.py \
		--source-checkpoint checkpoints/ctc_4090_65m_v2/last.ckpt \
		--tokenizer-model $(SPM_2K) \
		--target rnnt \
		--output checkpoints/rnnt_4090/encoder_from_ctc_v2.ckpt

average-checkpoints:
	cd $(REPO_ROOT) && $(UV) python scripts/average_checkpoints.py \
		--checkpoint-dir checkpoints/ctc_4090 \
		--last-n 5 \
		--output checkpoints/ctc_4090/averaged_last5.ckpt

# --- Ablations (smoke / comparison runs) ---

ablate-subsample-4x:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.ctc.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 10 \
		--checkpoint_dir checkpoints/ablate_subsample4x \
		--model.encoder.subsampling_factor 4 \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--wandb_run_name ablate-subsample-4x

ablate-kenlm-ctc:
	@echo "Hybrid vs E2E ablation: CTC+KenLM (hybrid LM) vs RNN-T greedy (E2E)."
	@echo "This compares system paradigms, not acoustic heads alone."
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.evaluate \
		--checkpoint checkpoints/ctc_4090/last.ckpt \
		--tokenizer_model $(SPM_4K) \
		--train_manifest $(TRAIN_MANIFEST) \
		--val_manifest $(VAL_MANIFEST) \
		--decode_types greedy beam_kenlm \
		--kenlm_model $(KENLM_MODEL) \
		--output_csv results/eval/ablate_ctc_kenlm.csv
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.evaluate \
		--checkpoint checkpoints/rnnt_4090/last.ckpt \
		--tokenizer_model $(SPM_4K) \
		--train_manifest $(TRAIN_MANIFEST) \
		--val_manifest $(VAL_MANIFEST) \
		--model_type tdt \
		--decode_types greedy \
		--val_max_symbols_per_t 10 \
		--output_csv results/eval/ablate_rnnt_greedy.csv

ablate-rnnt-clamp:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 5 \
		--checkpoint_dir checkpoints/ablate_rnnt_clamp_pos \
		--rnnt_clamp 1.0 \
		--no-compute-eval-loss \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--wandb_run_name ablate-rnnt-clamp-1.0
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 5 \
		--checkpoint_dir checkpoints/ablate_rnnt_clamp_neg \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--data.loader.train_max_batch_duration $(BATCH_DURATION) \
		--wandb_run_name ablate-rnnt-clamp--1.0
