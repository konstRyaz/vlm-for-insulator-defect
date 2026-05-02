from src.utils.init_utils import set_random_seed as _set_random_seed
from src.utils.init_utils import set_worker_seed


def set_random_seed(seed: int) -> None:
    _set_random_seed(int(seed))


def set_seed(seed: int) -> None:
    # Backward-compatible alias used by older imports.
    _set_random_seed(int(seed))


__all__ = ["set_random_seed", "set_seed", "set_worker_seed"]
