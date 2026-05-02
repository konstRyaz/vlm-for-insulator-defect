from __future__ import annotations

import logging
from typing import Any


class Writer:
    """No-op writer compatible with template-like API."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        project_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        **_: Any,
    ) -> None:
        self.logger = logger
        # Keep backward compatibility with old `config` keyword.
        self.project_config = project_config if project_config is not None else (config or {})
        self.config = self.project_config
        self.step = 0
        self.mode = "train"
        self.run_name = run_name

    def set_step(self, step: int, mode: str = "train") -> None:
        self.step = int(step)
        self.mode = mode

    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        _ = scalar_name
        _ = scalar

    def add_image(self, image_name: str, image: Any) -> None:
        _ = image_name
        _ = image

    def close(self) -> None:
        return
