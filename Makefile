UV_DEV := uv run --extra dev
UV := uv run

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TRAIN_MODULE := SpeechToText.models.ctc_attention.train

.PHONY: fmt download_dataset test types

fmt:
	cd $(REPO_ROOT) && $(UV_DEV) ruff format src scripts
	cd $(REPO_ROOT) && $(UV_DEV) ruff check src scripts --fix

download_dataset:
	cd $(REPO_ROOT) && $(UV) scripts/prepare_multilingual_asr.py
	cd $(REPO_ROOT) && $(UV) scripts/prepare_librispeech_clean_100.py

test:
	cd $(REPO_ROOT) && \
	$(UV) python -m $(TRAIN_MODULE) \
		--data.train_manifest data/debug/en_one.jsonl \
		--data.val_manifest data/debug/en_one.jsonl \
		--data.tokenizer_model models/sp_en_pl_balanced_unigram_2k_lower.model \
		--data.train_batch_size 1 \
		--data.val_batch_size 1 \
		--data.max_duration 25.0 \
		--max_epochs 180 \
		--model.d_model 128 \
		--model.n_layers 2 \
		--model.num_heads 2 \
		--optim.learning_rate 3e-4 \
		--precision 32-true \
		--wandb_run_name debug-overfit-one

types:
	cd $(REPO_ROOT) && $(UV_DEV) mypy src
