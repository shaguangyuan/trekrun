from __future__ import annotations


class MetricComputationError(Exception):
    """Raised when a metric cannot be computed from the given segment / landmarks."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
