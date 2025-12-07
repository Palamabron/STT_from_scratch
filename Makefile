UV_DEV := uv run --extra dev
UV := uv run

.PHONY: fmt download_dataset test typecheck

fmt:
	$(UV_DEV) ruff format src scripts
	$(UV_DEV) ruff check src scripts --fix

download_dataset:
	$(UV) scripts/prepare_multilingual_asr.py
	$(UV) scripts/prepare_librispeech_clean_100.py

test:
	$(UV) python -m src.SpeechToText.train \
		--data.train_manifest data/debug/en_one.jsonl \
		--data.val_manifest data/debug/en_one.jsonl \
		--data.tokenizer_model models/sp_en_pl_unigram_2k_lower.model \
		--data.train_batch_size 1 \
		--data.val_batch_size 1 \
		--data.max_duration 25.0 \
		--max_epochs 80 \
		--model.d_model 128 \
		--model.n_layers 2 \
		--model.num_heads 2 \
		--optim.learning_rate 5e-4 \
		--precision 32-true \
		--wandb_run_name debug-overfit-one

typecheck:
	$(UV_DEV) mypy src
