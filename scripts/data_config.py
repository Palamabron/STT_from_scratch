import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv  # Wymaga: pip install python-dotenv

# Ładowanie zmiennych z pliku .env (szuka w folderze skryptu lub wyżej)
load_dotenv()

# --- Ścieżki Projektu ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"

INDIVIDUAL_MANIFESTS_DIR = MANIFEST_DIR / "individual"
FINAL_MANIFEST_DIR = MANIFEST_DIR / "final"
FINAL_TRAIN_MANIFEST = FINAL_MANIFEST_DIR / "train_final.jsonl"
FINAL_VAL_MANIFEST = FINAL_MANIFEST_DIR / "val_final.jsonl"
FINAL_TEST_MANIFEST = FINAL_MANIFEST_DIR / "test_final.jsonl"

MODELS_DIR = ROOT_DIR / "models"
TOKENIZER_PREFIX = str(MODELS_DIR / "tokenizer_spm_bpe_1024")
TOKENIZER_CORPUS = DATA_DIR / "tokenizer_corpus.txt"

TARGET_SR = 16000

# --- Pobranie tokena z .env ---
# Upewnij się, że w pliku .env masz linię: HF_TOKEN=twój_token_tutaj
HF_TOKEN_VAL = os.getenv("HF_TOKEN")

if HF_TOKEN_VAL is None:
    print("⚠️  UWAGA: Nie znaleziono zmiennej HF_TOKEN w pliku .env ani w systemie.")
    print("    Niektóre zbiory (Common Voice, Bigos) mogą wymagać uwierzytelnienia.")


# --- Klasa Konfiguracyjna Datasetu ---
@dataclass
class DatasetConfig:
    name: str  # Unikalna nazwa (klucz w słowniku)
    hf_id: str  # ID z HuggingFace
    split: str  # Split (train, validation, test)
    lang: str  # Język (pl, en)
    samples: int  # Limit próbek
    text_col: str  # Nazwa kolumny z tekstem
    audio_col: str = "audio"  # Nazwa kolumny z audio (zazwyczaj "audio")
    config_name: str | None = None  # Config datasetu (np. "clean", "pl")

    # Flagi sterujące
    use_streaming: bool = True  # Czy używać streamingu (domyślnie tak)
    force_features: bool = False  # Czy aplikować fix dla Common Voice

    def __post_init__(self):
        # Automatyczna detekcja trybu szybkiego dla znanych mniejszych zbiorów
        fast_datasets = ["bigos", "pelcra", "fleurs", "mls_pl"]
        if any(x in self.hf_id for x in fast_datasets) or any(
            x in self.name for x in fast_datasets
        ):
            self.use_streaming = False  # Pobieramy całość na dysk (cache)

        # Automatyczna detekcja fixa dla Common Voice
        if "common_voice" in self.hf_id:
            self.force_features = True
            self.use_streaming = True  # CV musi być streamingiem (za duży)


# --- Definicje Zbiorów ---

# 1. Zbiory TRENINGOWE
TRAIN_DATASETS = [
    DatasetConfig(
        name="bigos_v2_train",
        hf_id="amu-cai/pl-asr-bigos-v2",
        config_name="pwr-azon_read-20",
        split="train",
        lang="pl",
        samples=28200,
        text_col="ref_orig",
    ),
    DatasetConfig(
        name="pelcra_pl_train",
        hf_id="pelcra/pl-asr-pelcra-for-bigos",
        config_name="ul-spokes_mix_luz-18",
        split="train",
        lang="pl",
        samples=5000,
        text_col="ref_orig",
    ),
    DatasetConfig(
        name="cv21_pl_train",
        hf_id="fsicoli/common_voice_21_0",
        config_name="pl",
        split="train",
        lang="pl",
        samples=14000,
        text_col="sentence",
    ),
    DatasetConfig(
        name="mls_pl_train",
        hf_id="facebook/multilingual_librispeech",
        config_name="polish",
        split="train",
        lang="pl",
        samples=8000,
        text_col="transcript",
    ),
    DatasetConfig(
        name="librispeech_train",
        hf_id="openslr/librispeech_asr",
        config_name="clean",
        split="train.360",
        lang="en",
        samples=38000,
        text_col="text",
    ),
    DatasetConfig(
        name="cv21_en_train",
        hf_id="fsicoli/common_voice_21_0",
        config_name="en",
        split="train",
        lang="en",
        samples=18000,
        text_col="sentence",
    ),
    DatasetConfig(
        name="fleurs_pl_train",
        hf_id="google/fleurs",
        config_name="pl_pl",
        split="train",
        lang="pl",
        samples=5000,
        text_col="transcription",
    ),
]

# 2. Zbiory WALIDACYJNE
VAL_DATASETS = [
    DatasetConfig(
        name="bigos_pl_clean_val",
        hf_id="amu-cai/pl-asr-bigos-v2",
        config_name="pwr-azon_read-20",
        split="validation",
        lang="pl",
        samples=2500,
        text_col="ref_orig",
    ),
    DatasetConfig(
        name="bigos_pl_noisy_val",
        hf_id="amu-cai/pl-asr-bigos-v2",
        config_name="pwr-azon_spont-20",
        split="train",  # Używamy TRAIN (test jest pusty)
        lang="pl",
        samples=3500,
        text_col="ref_orig",
    ),
    DatasetConfig(
        name="librispeech_val",
        hf_id="openslr/librispeech_asr",
        config_name="clean",
        split="validation",
        lang="en",
        samples=2500,
        text_col="text",
    ),
    DatasetConfig(
        name="cv21_pl_val",
        hf_id="fsicoli/common_voice_21_0",
        config_name="pl",
        split="validation",
        lang="pl",
        samples=2000,
        text_col="sentence",
    ),
    DatasetConfig(
        name="cv21_en_val",
        hf_id="fsicoli/common_voice_21_0",
        config_name="en",
        split="validation",
        lang="en",
        samples=2000,
        text_col="sentence",
    ),
    DatasetConfig(
        name="fleurs_pl_val",
        hf_id="google/fleurs",
        config_name="pl_pl",
        split="validation",
        lang="pl",
        samples=750,
        text_col="transcription",
    ),
    DatasetConfig(
        name="fleurs_en_val",
        hf_id="google/fleurs",
        config_name="en_us",
        split="validation",
        lang="en",
        samples=750,
        text_col="transcription",
    ),
    DatasetConfig(
        name="pelcra_pl_val",
        hf_id="pelcra/pl-asr-pelcra-for-bigos",
        config_name="ul-spokes_mix_luz-18",
        split="validation",
        lang="pl",
        samples=1500,
        text_col="ref_orig",
    ),
]

VOCAB_SIZE = 1024
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 1.0
