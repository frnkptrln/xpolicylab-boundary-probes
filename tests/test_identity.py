import numpy as np
import pytest

from boundary_probes.fixtures import observation, observation_batch
from boundary_probes.identity import IdentityProbeModel, semantic_fingerprint

CONFIG = {"env_cfg_type": "arx_x5", "action_type": "joint", "chunk_size": 2}


def test_requires_observation_before_single_action():
    with pytest.raises(RuntimeError):
        IdentityProbeModel(CONFIG).get_action()


def test_single_action_shape_and_nonzero_values():
    model = IdentityProbeModel(CONFIG)
    model.update_obs(observation())
    actions = model.get_action()

    assert len(actions) == 2
    assert set(actions[0]) == {
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    }
    assert actions[0]["left_arm_joint_state"].shape == (6,)
    assert actions[0]["left_ee_joint_state"].shape == (1,)
    assert np.all(actions[0]["left_arm_joint_state"] != 0)


def test_end_effector_mode_has_pose_dimensions():
    model = IdentityProbeModel({**CONFIG, "action_type": "ee"})
    model.update_obs(observation())
    action = model.get_action()[0]
    assert action["left_ee_pose"].shape == (7,)
    assert action["right_ee_pose"].shape == (7,)


def test_batch_rejects_duplicate_environment_identity():
    model = IdentityProbeModel(CONFIG)
    with pytest.raises(ValueError, match="duplicate env_idx"):
        model.update_obs_batch([observation(env_idx=2), observation(env_idx=2)])


def test_batch_without_embedded_ids_aligns_by_list_position():
    observations = [observation(env_idx=4), observation(env_idx=9)]
    for obs in observations:
        del obs["env_idx"]
    model = IdentityProbeModel(CONFIG)
    model.update_obs_batch(observations)
    actions = model.get_action_batch([4, 9])
    first = [float(chunk[0]["left_arm_joint_state"][0]) for chunk in actions]
    assert first[1] - first[0] == pytest.approx(0.005, abs=1e-7)


def test_batch_rejects_mixed_identity_presence():
    with_id = observation(env_idx=2)
    without_id = observation(env_idx=3)
    del without_id["env_idx"]
    model = IdentityProbeModel(CONFIG)
    with pytest.raises(ValueError, match="for every observation or for none"):
        model.update_obs_batch([with_id, without_id])


def test_batch_uses_requested_environment_order():
    observations = observation_batch(4)
    model = IdentityProbeModel(CONFIG)
    model.update_obs_batch(observations)
    requested = [3, 1, 2, 0]
    actions = model.get_action_batch(requested)
    first = [float(chunk[0]["left_arm_joint_state"][0]) for chunk in actions]

    expected = []
    for env_idx in requested:
        single = IdentityProbeModel(CONFIG)
        single.update_obs(observations[env_idx])
        expected.append(float(single.get_action()[0]["left_arm_joint_state"][0]))
    assert first == expected


def test_semantic_fingerprint_ignores_mapping_insertion_order():
    obs = observation()
    reordered = {**obs, "vision": dict(reversed(list(obs["vision"].items())))}
    assert semantic_fingerprint(obs) == semantic_fingerprint(reordered)


def test_reset_removes_stored_observation():
    model = IdentityProbeModel(CONFIG)
    model.update_obs(observation())
    model.reset()
    with pytest.raises(RuntimeError):
        model.get_action()
