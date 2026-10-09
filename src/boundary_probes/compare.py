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


def action_structure(value: Any) -> tuple:
    """Describe named fields, chunk nesting and numeric leaf shapes.

    Mapping order is irrelevant. Numeric lists and NumPy arrays of the same
    shape are equivalent; flattening must not erase a field or axis change.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("action field names must be strings")
        return ("mapping", tuple((key, action_structure(value[key])) for key in sorted(value)))
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("action leaves must be real numeric values, not text")
    try:
        array = np.asarray(value)
    except ValueError:
        array = None
    if array is not None and array.dtype.kind in "iuf":
        return ("numeric", tuple(array.shape))
    if isinstance(value, Sequence):
        return ("sequence", tuple(action_structure(item) for item in value))
    raise ValueError("action leaves must be real numeric values")


def action_delta(left: Any, right: Any) -> dict[str, Any]:
    same_structure = action_structure(left) == action_structure(right)
    left_flat = flatten_actions(left)
    right_flat = flatten_actions(right)
    finite = bool(np.isfinite(left_flat).all() and np.isfinite(right_flat).all())
    if not same_structure or left_flat.shape != right_flat.shape or not finite:
        return {
            "same_shape": same_structure and left_flat.shape == right_flat.shape,
            "same_structure": same_structure,
            "finite": finite,
            "left_size": int(left_flat.size),
            "right_size": int(right_flat.size),
            "max_abs": None,
        }
    # Finite operands can still overflow their difference. Keep JSON finite.
    with np.errstate(over="ignore", invalid="ignore"):
        maximum = float(np.max(np.abs(left_flat - right_flat))) if left_flat.size else 0.0
    return {
        "same_shape": True,
        "same_structure": True,
        "finite": finite,
        "left_size": int(left_flat.size),
        "right_size": int(right_flat.size),
        "max_abs": maximum if np.isfinite(maximum) else None,
    }


def within(delta: Mapping[str, Any], atol: float) -> bool:
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    return bool(
        delta["same_shape"]
        and delta.get("same_structure", True)
        and delta.get("finite", True)
        and delta["left_size"] > 0
        and delta["max_abs"] is not None
        and np.isfinite(delta["max_abs"])
        and delta["max_abs"] <= atol
    )
