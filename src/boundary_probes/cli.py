"""Command-line entry point for local boundary controls."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .contract import run_contract_probe
from .observation_perturbations import run_observation_probe
from .reset_leakage import run_reset_probe

UPSTREAM_REVISION = "c07a09614dd44cc4a67483bcb9a82e7439d99926"


def _record(command: str) -> dict[str, Any]:
    results = []
    if command in {"contract", "all"}:
        results.append(run_contract_probe())
    if command in {"reset", "all"}:
        results.append(run_reset_probe())
    if command in {"observation", "all"}:
        results.append(run_observation_probe())
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "upstream_revision": UPSTREAM_REVISION,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "status": "pass"
        if all(result["status"] == "pass" for result in results)
        else "fail",
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("contract", "reset", "observation", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--output", type=Path, help="optional JSON run-record path")
    args = parser.parse_args(argv)

    record = _record(args.command)
    rendered = json.dumps(record, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
