"""Small deterministic observations with XPolicyLab-compatible field names."""

from __future__ import annotations

from typing import Any

import numpy as np


def observation(env_idx: int = 0, variant: int = 0) -> dict[str, Any]:
    height, width = 8, 10
    grid = np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3)

    def image(offset: int) -> np.ndarray:
        return ((grid + offset + variant * 3) % 256).astype(np.uint8)

    joint = np.linspace(-0.3, 0.3, 6, dtype=np.float32) + np.float32(variant * 0.01)
    return {
        "data_format_version": "boundary-probe-v1",
        "instruction": f"place object variant {variant}",
        "env_idx": env_idx,
        "vision": {
            "cam_head": {"color": image(0)},
            "cam_left_wrist": {"color": image(17)},
            "cam_right_wrist": {"color": image(31)},
        },
        "state": {
            "left_arm_joint_state": joint,
            "left_ee_joint_state": np.array([0.25 + variant * 0.01], dtype=np.float32),
            "right_arm_joint_state": -joint,
            "right_ee_joint_state": np.array([0.75 - variant * 0.01], dtype=np.float32),
        },
    }


def observation_batch(size: int = 10, variant: int = 0) -> list[dict[str, Any]]:
    return [observation(env_idx=env_idx, variant=variant) for env_idx in range(size)]
