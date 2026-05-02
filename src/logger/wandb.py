from __future__ import annotations

import logging
from typing import Any

from src.logger.writer import Writer


class WandBWriter(Writer):
    def __init__(
        self,
        logger: logging.Logger,
        project_name: str,
        project_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        entity: str | None = None,
        mode: str = "online",
        run_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        cfg_dict = project_config if project_config is not None else (config or {})
        super().__init__(
            logger=logger,
            project_config=cfg_dict,
            run_name=run_name,
            **kwargs,
        )
        self.enabled = False
        self.wandb = None

        try:
            import wandb  # type: ignore

            self.wandb = wandb
            self.wandb.init(
                project=project_name,
                entity=entity,
                mode=mode,
                name=run_name,
                config=cfg_dict,
            )
            self.enabled = True
            logger.info("WandB writer enabled")
        except Exception as exc:  # pragma: no cover
            logger.warning("WandB is unavailable, fallback to no-op writer: %s", exc)

    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        if self.enabled and self.wandb is not None:
            self.wandb.log({f"{self.mode}/{scalar_name}": scalar}, step=self.step)

    def add_image(self, image_name: str, image: Any) -> None:
        if self.enabled and self.wandb is not None:
            self.wandb.log({image_name: self.wandb.Image(image)}, step=self.step)

    def close(self) -> None:
        if self.enabled and self.wandb is not None:
            self.wandb.finish()
