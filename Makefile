UV_DEV := uv run --extra dev
UV := uv run

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TRAIN_MANIFEST := data/manifests/final/train_final.jsonl
VAL_MANIFEST := data/manifests/final/val_final.jsonl
SPM_4K := models/spm_unigram_4k_trainval.model
SPM_8K := models/spm_unigram_8k_trainval_balanced.model

TRAIN_PATHS := \
	--data.manifests.train $(TRAIN_MANIFEST) \
	--data.manifests.val $(VAL_MANIFEST) \
	--data.tokenizer_model $(SPM_4K)

.PHONY: fmt prepare-data rebuild-manifests train-tokenizer train-tokenizer-8k \
	tokenizer-coverage test smoke-train types \
	train-ctc-4090 train-rnnt-4090 train-ctc-attn-4090 train-tdt-4090 \
	train-ctc-4090-oom train-rnnt-4090-oom train-ctc-attn-4090-oom \
	init-rnnt-from-ctc average-checkpoints \
	ablate-subsample-4x ablate-kenlm-ctc ablate-rnnt-clamp

fmt:
	cd $(REPO_ROOT) && $(UV_DEV) ruff format src scripts tests
	cd $(REPO_ROOT) && $(UV_DEV) ruff check src scripts tests --fix

prepare-data:
	cd $(REPO_ROOT)/scripts && $(UV) python -m prepare_data --config ../configs/data.yaml

rebuild-manifests:
	cd $(REPO_ROOT) && $(UV) python scripts/rebuild_final_manifests.py

train-tokenizer:
	cd $(REPO_ROOT) && $(UV) python scripts/train_tokenizer.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--corpus-out data/corpus/sp_trainval.txt \
		--model-prefix models/spm_unigram_4k_trainval \
		--vocab-size 4096 \
		--model-type unigram

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
		--model-path $(SPM_8K).model \
		--output-json results/tokenizer_coverage_8k.json

tokenizer-coverage:
	cd $(REPO_ROOT) && $(UV) python scripts/tokenizer_coverage_report.py \
		--manifests $(TRAIN_MANIFEST) $(VAL_MANIFEST) \
		--model-path $(SPM_4K) \
		--output-json results/tokenizer_coverage_4k.json

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
		--data.loader.train_max_batch_duration 1600 \
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
		--data.loader.train_max_batch_duration 1200 \
		--ctc_label_smoothing 0.1 \
		--aux_ctc_weight 0.3 \
		--spec_augment_start_epoch 16 \
		--audio_augment_start_epoch 7 \
		--optimizer.lr 2e-3 \
		--wandb_run_name ctc-4090-oom

train-rnnt-4090:
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 100 \
		--checkpoint_dir checkpoints/rnnt_4090 \
		--model.encoder.d_model 384 \
		--model.encoder.n_layers 16 \
		--model.encoder.n_heads 8 \
		--data.loader.train_max_batch_duration 1200 \
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
		--data.loader.train_max_batch_duration 800 \
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
		--data.loader.train_max_batch_duration 600 \
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
		--data.loader.train_max_batch_duration 400 \
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
		--data.loader.train_max_batch_duration 1200 \
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
		--data.loader.train_max_batch_duration 800 \
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
		--kenlm_model lm/pl_5gram.arpa \
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
		--data.loader.train_max_batch_duration 800 \
		--wandb_run_name ablate-rnnt-clamp-1.0
	cd $(REPO_ROOT) && $(UV) python -m SpeechToText.models.tdt.train \
		$(TRAIN_PATHS) \
		--precision bf16-mixed \
		--max_epochs 5 \
		--checkpoint_dir checkpoints/ablate_rnnt_clamp_neg \
		--rnnt_clamp -1.0 \
		--no-compute-eval-loss \
		--data.loader.train_max_batch_duration 800 \
		--wandb_run_name ablate-rnnt-clamp--1.0
