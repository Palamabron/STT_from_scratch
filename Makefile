UV_DEV := uv run --extra dev
UV := uv run

.PHONY: fmt download_dataset

fmt:
	$(UV_DEV) ruff format src scripts
	$(UV_DEV) ruff check src scripts --fix

download_dataset:
	$(UV) scripts/prepare_multilingual_asr.py
	$(UV) scripts/prepare_librispeech_clean_100.py
