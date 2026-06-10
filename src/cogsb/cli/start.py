from __future__ import annotations

from cogsb.cli.main import analyze


def main() -> None:
    """Start the live real-time analysis pipeline with default settings."""
    analyze(
        source="live",
        source_id=0,
        mode="realtime",
        output_dir="outputs",
        smoothness=0.35,
    )


__all__ = ["main"]
