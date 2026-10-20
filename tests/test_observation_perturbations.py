import json

import numpy as np
import pytest

from boundary_probes.cli import _record
from boundary_probes.fixtures import observation
from boundary_probes.identity import semantic_fingerprint
from boundary_probes.observation_perturbations import (
    SENSITIVITY_CASES,
    VisionBlindCanaryModel,
    run_observation_matrix,
    run_observation_probe,
)


def test_observation_probe_separates_invariance_from_sensitivity():
    result = run_observation_probe()

    assert result["status"] == "pass"
    assert result["checks"]["camera_mapping_order"]["observed"] == "invariant"
    assert result["checks"]["identical_frame_redelivery"]["observed"] == "invariant"
    for name in SENSITIVITY_CASES:
        assert result["checks"][name]["observed"] == "detectable"


def test_known_blind_canary_fails_only_sensitivity_controls():
    result = run_observation_matrix(VisionBlindCanaryModel)
    failed = sorted(
        name for name, check in result["checks"].items() if check["status"] == "fail"
    )

    assert result["status"] == "fail"
    assert failed == sorted(SENSITIVITY_CASES)


def test_latency_is_deferred_to_integration_boundary():
    result = run_observation_probe()
    latency = result["deferred"]["delivery_latency"]

    assert latency["status"] == "not_run"
    assert latency["required_evidence"] == "official_debug_client_timing"


def test_all_command_includes_stage_02():
    record = _record("all")

    assert record["status"] == "pass"
    assert [result["probe"] for result in record["results"]] == [
        "00_contract",
        "01_reset_leakage",
        "02_observation_perturbations",
        "03_action_transforms",
    ]


def policy_with_changed_action(changed_action, baseline_value=0.1):
    baseline = semantic_fingerprint(observation(env_idx=4, variant=0))

    class ChangedActionModel:
        def reset(self):
            self._obs = None

        def update_obs(self, obs):
            self._obs = obs

        def get_action(self):
            if semantic_fingerprint(self._obs) == baseline:
                return [{"arm_joint_state": np.array([baseline_value])}]
            return changed_action

    return ChangedActionModel


@pytest.mark.parametrize("invalid_action,baseline_value", [
    ([{"arm_joint_state": np.array([np.nan])}], 0.1),
    ([{"arm_joint_state": np.array([np.inf])}], 0.1),
    ([], 0.1),
    ({}, 0.1),
    ([{"arm_joint_state": np.array([])}], 0.1),
    ([{"arm_joint_state": np.array([0.2, 0.3])}], 0.1),
    ([{"wrong_arm": np.array([0.2])}], 0.1),
    ([{"arm_joint_state": np.array([1e308])}], -1e308),
], ids=["nan", "infinity", "empty-list", "empty-mapping", "empty-leaf",
        "wrong-shape", "renamed-field", "overflow-difference"])
def test_invalid_perturbed_actions_cannot_pass_as_detectable(
    invalid_action, baseline_value
):
    result = run_observation_matrix(
        policy_with_changed_action(invalid_action, baseline_value)
    )

    assert result["checks"]["camera_mapping_order"]["status"] == "pass"
    assert result["checks"]["identical_frame_redelivery"]["status"] == "pass"
    assert result["status"] == "fail"
    for name in SENSITIVITY_CASES:
        assert result["checks"][name]["status"] == "fail"
        assert result["checks"][name]["observed"] == "invalid"
    json.dumps(result, allow_nan=False)


def test_valid_changed_actions_remain_detectable():
    changed = [{"arm_joint_state": np.array([0.2])}]
    result = run_observation_matrix(policy_with_changed_action(changed))

    assert result["status"] == "pass"
    for name in SENSITIVITY_CASES:
        assert result["checks"][name]["observed"] == "detectable"
