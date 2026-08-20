"""Comparison helpers shared by contract and reset probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def flatten_actions(actions: Any) -> np.ndarray:
    values: list[np.ndarray] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                visit(item)
        else:
            values.append(np.asarray(value, dtype=np.float64).reshape(-1))

    visit(actions)
    return np.concatenate(values) if values else np.array([], dtype=np.float64)


def action_delta(left: Any, right: Any) -> dict[str, Any]:
    left_flat = flatten_actions(left)
    right_flat = flatten_actions(right)
    if left_flat.shape != right_flat.shape:
        return {
            "same_shape": False,
            "left_size": int(left_flat.size),
            "right_size": int(right_flat.size),
            "max_abs": None,
        }
    maximum = float(np.max(np.abs(left_flat - right_flat))) if left_flat.size else 0.0
    return {
        "same_shape": True,
        "left_size": int(left_flat.size),
        "right_size": int(right_flat.size),
        "max_abs": maximum,
    }


def within(delta: Mapping[str, Any], atol: float) -> bool:
    return bool(
        delta["same_shape"]
        and delta["max_abs"] is not None
        and delta["max_abs"] <= atol
    )
