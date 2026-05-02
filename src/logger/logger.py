import json
import logging
import logging.config
from pathlib import Path


def setup_logging(save_dir: Path, append: bool = False) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).resolve().parent / "logger_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    file_mode = "a" if append else "w"
    config["handlers"]["file"]["filename"] = str(save_dir / "train.log")
    config["handlers"]["file"]["mode"] = file_mode

    logging.config.dictConfig(config)


def get_logger(name: str = "train") -> logging.Logger:
    return logging.getLogger(name)
