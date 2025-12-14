import io
import json
import logging
from pathlib import Path

import datasets
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Features, Value
from tqdm import tqdm

# Import config
import data_config
from data_config import DatasetConfig

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wyciszanie bibliotek
for lib in ["httpx", "urllib3", "fsspec", "datasets"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# --- MONKEY PATCH ---
try:
    original_init = datasets.DatasetInfo.__init__

    def patched_init(self, *args, **kwargs):
        if "task_templates" in kwargs:
            del kwargs["task_templates"]
        original_init(self, *args, **kwargs)

    datasets.DatasetInfo.__init__ = patched_init
    logger.info("✅ Applied datasets.DatasetInfo patch.")
except Exception as e:
    logger.warning(f"⚠️ Patch failed: {e}")


# --- COMMON VOICE FEATURES ---
def get_cv_features():
    return Features(
        {
            "client_id": Value("string"),
            "path": Value("string"),
            "audio": Audio(decode=False),
            "sentence": Value("string"),
            "up_votes": Value("string"),
            "down_votes": Value("string"),
            "age": Value("string"),
            "gender": Value("string"),
            "accent": Value("string"),
            "locale": Value("string"),
            "segment": Value("string"),
            "variant": Value("string"),
            "sentence_id": Value("string"),
            "sentence_domain": Value("string"),
        }
    )


def process_dataset(ds_config: DatasetConfig, output_dir: Path):
    """Główna funkcja przetwarzająca pojedynczy dataset."""
    manifest_path = output_dir / f"{ds_config.name}.jsonl"

    if manifest_path.exists():
        logger.info(f"⏭️  Manifest {manifest_path.name} exists. Skipping.")
        return

    mode_str = "FAST (Cache)" if not ds_config.use_streaming else "STREAMING"
    logger.info(f"🚀 Processing: {ds_config.name} | Mode: {mode_str} | Limit: {ds_config.samples}")

    load_kwargs = {
        "path": ds_config.hf_id,
        "name": ds_config.config_name,
        "split": ds_config.split,
        "streaming": ds_config.use_streaming,
        "trust_remote_code": True,
        "token": data_config.HF_TOKEN_VAL,
    }

    if ds_config.force_features:
        logger.info(f"   🔧 Applying custom features fix for {ds_config.name}")
        load_kwargs["features"] = get_cv_features()

    try:
        ds = datasets.load_dataset(**load_kwargs)
        if not ds_config.use_streaming:
            ds = ds.shuffle(seed=42)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR loading {ds_config.name}: {e}")
        return

    # Przygotowanie folderu na audio
    audio_output_dir = data_config.AUDIO_DIR / ds_config.name
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    data_list = []
    count = 0
    iterator = iter(ds)
    pbar = tqdm(total=ds_config.samples, desc=f"Downloading {ds_config.name}")

    while count < ds_config.samples:
        try:
            sample = next(iterator)
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"⚠️ Iteration error in {ds_config.name}: {e}")
            continue

        try:
            # 1. Pobranie i weryfikacja tekstu
            text = sample.get(ds_config.text_col)
            if not text or len(str(text).strip()) < 2:
                continue

            # 2. Dekodowanie Audio
            if ds_config.force_features:  # CV (mp3 bytes)
                audio_bytes = sample[ds_config.audio_col]["bytes"]
                audio_array, orig_sr = sf.read(io.BytesIO(audio_bytes))
            else:  # Standard (decoded array)
                audio_info = sample[ds_config.audio_col]
                audio_array = audio_info["array"]
                orig_sr = audio_info["sampling_rate"]

            # 3. Przetwarzanie (Resample + Normalize)
            tensor_wav = torch.tensor(audio_array, dtype=torch.float32)
            if tensor_wav.ndim == 1:
                tensor_wav = tensor_wav.unsqueeze(0)
            elif tensor_wav.shape[0] > tensor_wav.shape[1]:
                tensor_wav = tensor_wav.t()

            if orig_sr != data_config.TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_sr, data_config.TARGET_SR)
                tensor_wav = resampler(tensor_wav)

            max_val = torch.max(torch.abs(tensor_wav))
            if max_val > 0:
                tensor_wav = tensor_wav / (max_val + 1e-6)

            # 4. Zapis WAV (Soundfile)
            filename = f"{ds_config.name}_{count:06d}.wav"
            wav_path = audio_output_dir / filename
            sf.write(str(wav_path), tensor_wav.squeeze().numpy(), data_config.TARGET_SR)

            # 5. Dodanie do listy
            duration = tensor_wav.shape[-1] / data_config.TARGET_SR
            data_list.append(
                {
                    "audio_filepath": str(wav_path.resolve()),
                    "text": str(text).strip().lower(),
                    "duration": float(duration),
                    "language": ds_config.lang,
                    "dataset": ds_config.name,
                }
            )

            count += 1
            pbar.update(1)

        except Exception:
            # logger.warning(f"Sample error: {e}")
            continue  # Silent continue for speed

    pbar.close()

    # Zapis Manifestu
    if data_list:
        with open(manifest_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item) + "\n")
        logger.info(f"✅ Saved manifest: {manifest_path} ({len(data_list)} samples)")
    else:
        logger.warning(f"⚠️ Empty manifest for {ds_config.name}!")

    return manifest_path if data_list else None


def merge_manifests(manifest_files, output_file):
    logger.info(f"🔄 Merging {len(manifest_files)} manifests -> {output_file}")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for m_file in manifest_files:
            if m_file and m_file.exists():
                with open(m_file, encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
    logger.info("✅ Merge complete.")


def train_tokenizer(manifest_path, vocab_size):
    logger.info(f"🔨 Training Tokenizer (BPE, vocab={vocab_size})...")
    corpus_file = data_config.TOKENIZER_CORPUS

    with (
        open(manifest_path, encoding="utf-8") as f_in,
        open(corpus_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data = json.loads(line)
            f_out.write(data["text"] + "\n")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=data_config.TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type=data_config.MODEL_TYPE,
        character_coverage=data_config.CHARACTER_COVERAGE,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )
    logger.info(f"✅ Tokenizer saved: {data_config.TOKENIZER_PREFIX}.model")


def main():
    # Setup
    for p in [
        data_config.INDIVIDUAL_MANIFESTS_DIR,
        data_config.FINAL_MANIFEST_DIR,
        data_config.MODELS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Train
    logger.info("--- 🟢 STARTING TRAIN DATASETS ---")
    train_manifests = []
    for ds_conf in data_config.TRAIN_DATASETS:
        m_path = process_dataset(ds_conf, data_config.INDIVIDUAL_MANIFESTS_DIR)
        if m_path:
            train_manifests.append(m_path)

    # 2. Validation
    logger.info("--- 🔵 STARTING VALIDATION DATASETS ---")
    val_manifests = []
    for ds_conf in data_config.VAL_DATASETS:
        m_path = process_dataset(ds_conf, data_config.INDIVIDUAL_MANIFESTS_DIR)
        if m_path:
            val_manifests.append(m_path)

    # 3. Finalize
    if train_manifests:
        merge_manifests(train_manifests, data_config.FINAL_TRAIN_MANIFEST)
        train_tokenizer(data_config.FINAL_TRAIN_MANIFEST, data_config.VOCAB_SIZE)

    if val_manifests:
        merge_manifests(val_manifests, data_config.FINAL_VAL_MANIFEST)

    logger.info("=== 🏁 DATA PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()
