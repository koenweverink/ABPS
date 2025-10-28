"""Lightweight logging utilities.

The original implementation always wrote INFO level messages to
``simulation.log`` which incurred a noticeable overhead when the genetic
algorithm evaluated many simulations.  To speed things up we make logging
optional and disabled by default.  Set the environment variable
``SIM_LOG_LEVEL`` (e.g. ``INFO`` or ``DEBUG``) to enable file logging.  Any
other value, including the default ``OFF``, results in a no-op logger.
"""

from __future__ import annotations

import logging
import os


class _NullLogger:
    """A logger that silently ignores all messages."""

    def __getattr__(self, _name):  # pragma: no cover - trivial
        def _noop(*args, **kwargs):
            pass

        return _noop


# LOG_LEVEL = os.getenv("SIM_LOG_LEVEL", "OFF").upper()
LOG_LEVEL = "ON"

if LOG_LEVEL == "OFF":
    logger = _NullLogger()
else:
    logging.basicConfig(
        filename="simulation.log",
        filemode="w",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger("Sim")

