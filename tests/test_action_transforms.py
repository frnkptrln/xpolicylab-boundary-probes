import numpy as np
import pytest
from boundary_probes.action_transforms import (
    decode_joints, decode_poses, encode_joints, encode_poses,
    fixtures, pose_error, run_action_probe,
)


def test_analytic_joint_and_pose_witnesses():
    joints = np.array([0, np.pi/2, np.pi, -np.pi/2, 0.5, -0.5])
    assert np.allclose(encode_joints(joints), [-180, 0, 90/np.pi, 90, 90/np.pi, 90])
    pose = [1., 0., 0., 0., 0., 0., 1.]
    assert np.allclose(encode_poses(pose), [0.3, 0.8, 0.5, 0, 0, np.sqrt(.5), np.sqrt(.5)])
    assert np.allclose(decode_poses(encode_poses(pose)), pose)


def test_round_trip_and_input_immutability():
    joints, poses = fixtures()
    j0, p0 = joints.copy(), poses.copy()
    assert np.allclose(decode_joints(encode_joints(joints)), joints)
    assert max(pose_error(decode_poses(encode_poses(poses)), poses).values()) < 1e-10
    assert np.array_equal(j0, joints) and np.array_equal(p0, poses)


def test_rotation_comparison_ignores_quaternion_sign_but_not_axis():
    p = np.array([0.,0.,0.,1.,0.,0.,0.])
    flipped = p.copy(); flipped[3:] *= -1
    assert pose_error(p, flipped)["orientation_max_chord"] == 0
    other = np.array([0.,0.,0.,0.,1.,0.,0.])
    assert pose_error(p, other)["orientation_max_chord"] > 1


@pytest.mark.parametrize('bad', [np.zeros(7), [0,0,0,0,0,0,np.nan], np.zeros(6), np.empty((0,7))])
def test_invalid_poses_rejected(bad):
    with pytest.raises(ValueError): encode_poses(bad)


def test_matrix_and_canaries():
    result = run_action_probe()
    assert result["status"] == "pass"
    assert len(result["checks"]) == 12
    assert all(v["status"] == "pass" for k,v in result["checks"].items() if k.startswith("canary_"))


def test_broken_decoder_fails_required_control(monkeypatch):
    import boundary_probes.action_transforms as module
    monkeypatch.setattr(module, "decode_joints", lambda x: np.deg2rad(x))
    result = module.run_action_probe()
    assert result["status"] == "fail"
    assert result["checks"]["joint_round_trip"]["status"] == "fail"


def test_action_cli_includes_stage03():
    from boundary_probes.cli import _record
    assert _record("action")["results"][0]["probe"] == "03_action_transforms"
    assert _record("all")["results"][-1]["probe"] == "03_action_transforms"
