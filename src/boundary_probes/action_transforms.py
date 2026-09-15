"""Stage 03: fixture-level coordinate equivalence, with faulty decoders.

Joint coordinates are radians in the canonical frame. Pose fixtures use xyz
in metres and unit quaternions xyzw, active rotations. These are explicit
probe conventions, not an assertion about every XPolicyLab controller.
"""
from __future__ import annotations

from typing import Any
import numpy as np

PERMUTATION = np.array([2, 0, 5, 1, 4, 3])
SIGNS = np.array([-1., 1., -1., 1., 1., -1.])
TRANSLATION = np.array([0.3, -0.2, 0.5])
FRAME_QUATERNION = np.array([0., 0., np.sqrt(0.5), np.sqrt(0.5)])
FRAME_ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def finite_array(value: Any, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError(f"expected finite coordinates with last dimension {width}")
    if array.size == 0:
        raise ValueError("empty trajectories are not an equivalence witness")
    return array


def encode_joints(canonical: Any) -> np.ndarray:
    """Canonical joint order/radians -> signed permuted order/degrees."""
    return np.rad2deg(finite_array(canonical, 6)[..., PERMUTATION] * SIGNS)


def decode_joints(encoded: Any) -> np.ndarray:
    ordered = np.deg2rad(finite_array(encoded, 6)) * SIGNS
    return ordered[..., np.argsort(PERMUTATION)]


def quaternion_product(a: Any, b: Any) -> np.ndarray:
    a, b = np.broadcast_arrays(finite_array(a, 4), finite_array(b, 4))
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)), axis=-1)


def unit_pose(value: Any) -> np.ndarray:
    pose = finite_array(value, 7)
    if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1., rtol=0., atol=1e-10):
        raise ValueError("pose quaternions must have unit norm (xyzw)")
    return pose


def encode_poses(canonical: Any) -> np.ndarray:
    pose = unit_pose(canonical)
    position = pose[..., :3] @ FRAME_ROTATION.T + TRANSLATION
    orientation = quaternion_product(FRAME_QUATERNION, pose[..., 3:])
    return np.concatenate((position, orientation), axis=-1)


def decode_poses(encoded: Any) -> np.ndarray:
    pose = unit_pose(encoded)
    position = (pose[..., :3] - TRANSLATION) @ FRAME_ROTATION
    inverse = FRAME_QUATERNION * np.array([-1., -1., -1., 1.])
    return np.concatenate((position, quaternion_product(inverse, pose[..., 3:])), axis=-1)


def pose_error(left: Any, right: Any) -> dict:
    left, right = unit_pose(left), unit_pose(right)
    if left.shape != right.shape:
        raise ValueError("trajectory shape/order must agree")
    a, b = left[..., 3:], right[..., 3:]
    # q and -q are the same rotation. Chord error avoids acos conditioning at 1.
    orientation = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return {"position_max_abs_m": float(np.max(np.abs(left[..., :3] - right[..., :3]))),
            "orientation_max_chord": float(np.max(orientation))}


def fixtures() -> tuple[np.ndarray, np.ndarray]:
    # Two environments, three ordered actions, distinct coordinates and poses.
    joint = (np.arange(36, dtype=float).reshape(2, 3, 6) - 17.5) / 25.
    angle = np.array([[0., 0.4, -0.6], [0.8, -1., 1.2]])
    poses = np.zeros((2, 3, 7))
    poses[..., :3] = (np.arange(18).reshape(2, 3, 3) - 8.5) / 10.
    poses[..., 3] = np.sin(angle / 2)
    poses[..., 6] = np.cos(angle / 2)
    return joint, poses


def run_action_probe(atol: float = 1e-10) -> dict:
    if not np.isfinite(atol) or not 0 < atol < 1e-4:
        raise ValueError("atol must be positive, finite and below 1e-4")
    joint, poses = fixtures()
    encoded_joints = encode_joints(joint)
    encoded_poses = encode_poses(poses)
    decoded_joints = decode_joints(encoded_joints)
    pose_delta = pose_error(poses, decode_poses(encoded_poses))
    sign_flipped = poses.copy()
    sign_flipped[..., 3:] *= -1
    sign_delta = pose_error(poses, sign_flipped)
    checks = {}
    def check(name, passed, **values):
        checks[name] = {"status": "pass" if passed else "fail", **values}
    joint_error = float(np.max(np.abs(joint - decoded_joints)))
    check("joint_round_trip", joint_error <= atol, canonical_max_abs_rad=joint_error)
    check("pose_round_trip", max(pose_delta.values()) <= atol, **pose_delta)
    check("quaternion_sign_equivalence", max(sign_delta.values()) <= atol, **sign_delta)
    check("joint_single_batch_parity", all(np.allclose(decode_joints(encode_joints(row)), decoded_joints[i], rtol=0., atol=atol) for i, row in enumerate(joint)))
    check("pose_single_batch_parity", all(max(pose_error(decode_poses(encode_poses(row)), decode_poses(encoded_poses)[i]).values()) <= atol for i, row in enumerate(poses)))
    check("nontrivial_joint_encoding", not np.allclose(joint, encoded_joints, rtol=0., atol=atol))
    check("nontrivial_pose_encoding", max(pose_error(poses, encoded_poses).values()) > atol)
    # Known faulty paths: equivalent encodings become observably wrong if the
    # decoder omits a transform or loses the trajectory/environment identity.
    wrong_order = np.deg2rad(encoded_joints) * SIGNS
    check("canary_missing_joint_permutation", float(np.max(np.abs(joint - wrong_order))) > atol)
    check("canary_missing_joint_units", float(np.max(np.abs(joint - encoded_joints))) > atol)
    no_translation = decode_poses(encoded_poses).copy()
    no_translation[..., :3] += TRANSLATION @ FRAME_ROTATION
    check("canary_missing_pose_translation", max(pose_error(poses, no_translation).values()) > atol)
    check("canary_swapped_environments", max(pose_error(poses, decode_poses(encoded_poses[::-1])).values()) > atol)
    check("canary_reversed_time", max(pose_error(poses, decode_poses(encoded_poses[:, ::-1])).values()) > atol)
    return {"probe": "03_action_transforms", "status": "pass" if all(c["status"] == "pass" for c in checks.values()) else "fail",
            "atol": atol, "fixture_shape": {"environments": 2, "actions_per_environment": 3},
            "conventions": {"joints": "canonical radians, encoded signed permutation in degrees",
                            "pose": "xyz metres + unit quaternion xyzw; active 90-degree z frame rotation then translation"},
            "checks": checks,
            "deferred": {"official_controller_integration": {"status": "not_run", "reason": "requires verified controller-specific action convention and official client/server trace"},
                         "learned_policy_divergence": {"status": "not_run", "reason": "Stage 04 requires a pinned learned policy and separate run authorization"}},
            "claim_boundary": "coordinate equivalence for explicit deterministic fixtures; no learned-policy, simulator, robot, timing or task-competence result"}
