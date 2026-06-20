UV_DEV := uv run --extra dev
UV := uv run

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TRAIN_MODULE := SpeechToText.models.ctc.train

.PHONY: fmt prepare-data rebuild-manifests train-tokenizer test smoke-train types

fmt:
	cd $(REPO_ROOT) && $(UV_DEV) ruff format src scripts
	cd $(REPO_ROOT) && $(UV_DEV) ruff check src scripts --fix

prepare-data:
	cd $(REPO_ROOT)/scripts && $(UV) python -m prepare_data --config ../configs/data.yaml

rebuild-manifests:
	cd $(REPO_ROOT) && $(UV) python scripts/rebuild_final_manifests.py

train-tokenizer:
	cd $(REPO_ROOT) && $(UV) python scripts/train_tokenizer.py \
		--manifests data/manifests/final/train_final.jsonl data/manifests/final/val_final.jsonl \
		--corpus-out data/corpus/sp_trainval.txt \
		--model-prefix models/spm_unigram_4k_trainval \
		--vocab-size 4096 \
		--model-type unigram

test:
	cd $(REPO_ROOT) && $(UV_DEV) pytest tests/

smoke-train:
	cd $(REPO_ROOT) && \
	$(UV) python -m $(TRAIN_MODULE) \
		--data.manifests.train data/debug/en_one.jsonl \
		--data.manifests.val data/debug/en_one.jsonl \
		--data.tokenizer_model models/spm_unigram_4k_trainval.model \
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
		--wandb_run_name debug-overfit-one

types:
	cd $(REPO_ROOT) && $(UV_DEV) mypy src
