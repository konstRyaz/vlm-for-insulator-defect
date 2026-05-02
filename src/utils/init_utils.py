import logging
import os
import random
import secrets
import string
import subprocess
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.logger.logger import setup_logging


def set_worker_seed(worker_id: int) -> None:
    _ = worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(
    save_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    commit_path = save_dir / "git_commit.txt"
    patch_path = save_dir / "git_diff.patch"

    if not (Path.cwd() / ".git").exists():
        if logger is not None:
            logger.warning("Git metadata skipped: .git not found in current workspace.")
        return

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_result.returncode != 0:
            if logger is not None:
                logger.warning("Git metadata skipped: `git rev-parse HEAD` failed.")
            return
        commit_path.write_text(commit_result.stdout.strip() + "\n", encoding="utf-8")

        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_result.returncode != 0:
            if logger is not None:
                logger.warning("Git metadata skipped: `git diff HEAD` failed.")
            return
        patch_path.write_text(diff_result.stdout, encoding="utf-8")
    except Exception as exc:
        if logger is not None:
            logger.warning("Git metadata skipped: %s", exc)


def setup_saving_and_logging(config) -> tuple[logging.Logger, Path]:
    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, hydra_dir / "resolved_config.yaml", resolve=True)

    log_git_commit_and_patch(run_dir, logger=logger)

    return logger, run_dir
