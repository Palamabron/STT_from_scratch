from __future__ import annotations

import argparse
from pathlib import Path

import tyro
import yaml
from dotenv import load_dotenv

from .config import AppConfig, DatasetSpec
from .pipeline import run_pipeline


def load_yaml_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    run = raw["run"]
    hf = raw.get("hf", {"token_env": "HF_TOKEN"})
    datasets_section = raw["datasets"]

    training = raw.get("training", {})

    train_specs = [DatasetSpec(**item) for item in datasets_section.get("train", [])]
    val_specs = [DatasetSpec(**item) for item in datasets_section.get("val", [])]

    return AppConfig(
        paths=type(AppConfig().paths)(**{k: Path(v) for k, v in paths.items()}),
        run=type(AppConfig().run)(**run),
        hf=type(AppConfig().hf)(**hf),
        datasets=type(AppConfig().datasets)(train=train_specs, val=val_specs),
        max_train_hours=training.get("max_train_hours"),
    )


def resolve_config_path(config_arg: str | None) -> Path:
    default_path = Path("configs/data.yaml")
    candidates: list[Path] = []

    if config_arg is None or str(config_arg).strip() == "":
        candidates.append(default_path)
    else:
        path = Path(config_arg)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(Path("configs") / path.name)
            candidates.append(Path("configs") / path)
            stem = path.stem
            candidates.append(Path("configs") / f"{stem}.yaml")
            candidates.append(Path("configs") / f"{stem}.yml")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = []
    cfg_dir = Path("configs")
    if cfg_dir.exists():
        available = sorted([path.name for path in cfg_dir.glob("*.y*ml")])

    raise FileNotFoundError(
        f"Config not found. Tried: {[str(c) for c in candidates]}. "
        f"Available in ./configs: {available}"
    )


def entrypoint() -> None:
    load_dotenv()
    argument_parser = argparse.ArgumentParser(add_help=False)
    argument_parser.add_argument("--config", type=str, required=False, default=None)
    parsed_args, remaining_args = argument_parser.parse_known_args()

    config_path = resolve_config_path(parsed_args.config)
    base_config = load_yaml_config(config_path)
    app_config = tyro.cli(AppConfig, default=base_config, args=remaining_args)
    run_pipeline(app_config)
