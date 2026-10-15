"""The exception type raised for every speakrs failure."""

from __future__ import annotations


class SpeakrsError(Exception):
    """A speakrs operation failed.

    ``category`` is one of the stable values below, so callers can branch on it
    without parsing messages. ``code`` is equal to ``category`` unless a failure
    needs a finer distinction, such as ``queue_full`` or ``queue_closed`` in the
    ``queue`` category.

    Categories: ``configuration``, ``unsupported_runtime``, ``model_prepare``,
    ``model_manifest``, ``model_load``, ``audio_decode``, ``audio_resample``,
    ``pipeline``, ``queue``, ``cancelled``, ``internal``.
    """

    def __init__(self, category: str, code: str, message: str) -> None:
        super().__init__(category, code, message)

    @property
    def category(self) -> str:
        """Broad, stable failure category."""
        return self.args[0]

    @property
    def code(self) -> str:
        """Finer, stable failure code."""
        return self.args[1]

    @property
    def message(self) -> str:
        """Human-readable description."""
        return self.args[2]

    def __str__(self) -> str:
        return f"{self.message} [{self.code}]"
