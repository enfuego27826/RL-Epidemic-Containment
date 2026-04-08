"""TensorBoard logger wrapper for PPO training diagnostics.

Provides a thin, optional wrapper around ``torch.utils.tensorboard.SummaryWriter``
that degrades gracefully to plain logging when TensorBoard is not available.

Usage
-----
::

    from src.train.tb_logger import TBLogger

    tb = TBLogger(log_dir="runs/baseline_seed42")
    tb.log_scalar("train/policy_loss", value, step)
    tb.log_scalars("train", {"policy_loss": pl, "value_loss": vl}, step)
    tb.close()

The logger can also be constructed from a config dict::

    tb = TBLogger.from_config(config, run_tag="baseline")

All log calls are no-ops when TensorBoard is unavailable, so training code
can unconditionally call ``tb.log_*`` without try/except guards.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import TensorBoard once at module load; flag availability.
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore[import]
    _TB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SummaryWriter = None  # type: ignore[assignment,misc]
    _TB_AVAILABLE = False
    logger.debug(
        "torch.utils.tensorboard not available — TBLogger will emit text logs only. "
        "Install tensorboard (`pip install tensorboard`) to enable visual dashboards."
    )


class TBLogger:
    """Thin wrapper around ``SummaryWriter`` with a text-log fallback.

    Parameters
    ----------
    log_dir:
        Directory where TensorBoard event files are written.
        Set to ``None`` or empty string to disable TensorBoard entirely.
    enabled:
        Explicit override to disable TensorBoard even when available.
    """

    def __init__(self, log_dir: str | None = None, enabled: bool = True) -> None:
        self._writer: Any = None
        self._log_dir = log_dir or "runs/default"
        self._enabled = enabled and _TB_AVAILABLE and bool(log_dir)

        if self._enabled:
            os.makedirs(self._log_dir, exist_ok=True)
            try:
                self._writer = _SummaryWriter(log_dir=self._log_dir)
                logger.info("TensorBoard logging to: %s", self._log_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create SummaryWriter: %s — disabling TBLogger.", exc)
                self._enabled = False
                self._writer = None
        else:
            if log_dir:
                logger.debug("TBLogger disabled (enabled=%s, tb_available=%s).", enabled, _TB_AVAILABLE)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any], run_tag: str = "") -> "TBLogger":
        """Construct a ``TBLogger`` from a training config dict.

        Reads ``config["logging"]["tensorboard_dir"]`` (default:
        ``"runs/<checkpoint_dir_basename>"``).  Set the key to ``null`` or
        ``false`` to disable TensorBoard.

        Parameters
        ----------
        config:
            Full experiment config dict.
        run_tag:
            Optional suffix appended to the log directory name for
            disambiguation when running multiple seeds.

        Returns
        -------
        TBLogger
        """
        logging_cfg = config.get("logging", {})

        # Respect explicit tb_enabled flag
        tb_enabled: bool = bool(logging_cfg.get("tensorboard_enabled", True))

        # Derive log directory
        tb_dir: str | None = logging_cfg.get("tensorboard_dir")
        if not tb_dir:
            ckpt_dir = logging_cfg.get("checkpoint_dir", "checkpoints/run")
            base = os.path.basename(ckpt_dir.rstrip("/"))
            tb_dir = os.path.join("runs", base)
        if run_tag:
            tb_dir = f"{tb_dir}_{run_tag}"

        return cls(log_dir=tb_dir, enabled=tb_enabled)

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar to TensorBoard and (optionally) to the Python logger."""
        if self._writer is not None:
            try:
                self._writer.add_scalar(tag, value, global_step=step)
            except Exception as exc:  # noqa: BLE001
                logger.debug("TBLogger.log_scalar failed: %s", exc)

    def log_scalars(self, group: str, metrics: dict[str, float], step: int) -> None:
        """Write multiple scalars under a common ``group/`` tag prefix.

        Parameters
        ----------
        group:
            Tag prefix, e.g. ``"train"`` → ``"train/policy_loss"``.
        metrics:
            Dict mapping metric name to float value.
        step:
            Global step counter.
        """
        for name, value in metrics.items():
            self.log_scalar(f"{group}/{name}", value, step)

    # ------------------------------------------------------------------
    # Histogram logging (optional, for advantage distributions etc.)
    # ------------------------------------------------------------------

    def log_histogram(self, tag: str, values: list[float], step: int) -> None:
        """Write a histogram of ``values`` to TensorBoard."""
        if self._writer is not None and values:
            try:
                import torch  # type: ignore[import]
                self._writer.add_histogram(tag, torch.tensor(values, dtype=torch.float32), step)
            except Exception as exc:  # noqa: BLE001
                logger.debug("TBLogger.log_histogram failed: %s", exc)

    # ------------------------------------------------------------------
    # Text logging
    # ------------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Write a markdown text string to TensorBoard."""
        if self._writer is not None:
            try:
                self._writer.add_text(tag, text, global_step=step)
            except Exception as exc:  # noqa: BLE001
                logger.debug("TBLogger.log_text failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """Close the underlying ``SummaryWriter`` and free resources."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None

    @property
    def log_dir(self) -> str:
        """The directory where TensorBoard event files are written."""
        return self._log_dir

    @property
    def is_active(self) -> bool:
        """True if TensorBoard is actually writing events."""
        return self._writer is not None

    def __repr__(self) -> str:
        return f"TBLogger(log_dir={self._log_dir!r}, active={self.is_active})"
