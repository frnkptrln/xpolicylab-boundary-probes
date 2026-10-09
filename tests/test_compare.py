import json

import numpy as np
import pytest

from boundary_probes.compare import action_delta, within


def test_identical_numbers_under_different_actuators_do_not_pass():
    delta = action_delta({"left_arm": [1, 2]}, {"right_arm": [1, 2]})
    assert delta["left_size"] == delta["right_size"] == 2
    assert not delta["same_structure"]
    assert not within(delta, 0)


def test_flattening_does_not_hide_axis_or_chunk_changes():
    for left, right in [
        (np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2)),
        ([{"arm": [1]}, {"arm": [2]}], [{"arm": [1, 2]}]),
        ({"arm": np.array(1)}, {"arm": np.array([1])}),
    ]:
        assert not within(action_delta(left, right), 0)


def test_equivalent_numeric_containers_and_mapping_order_pass():
    assert within(action_delta(
        {"left": [1, 2], "right": (3, 4)},
        {"right": np.array([3, 4]), "left": np.array([1., 2.])},
    ), 0)


def test_empty_and_nonfinite_actions_fail_without_nonstandard_json():
    for left, right in [([], []), ({}, {}), ([np.nan], [np.nan]),
                        ([np.inf], [np.inf]), ([-1e308], [1e308])]:
        delta = action_delta(left, right)
        assert not within(delta, 1)
        json.dumps(delta, allow_nan=False)


@pytest.mark.parametrize("value", ["1", None, True, 1 + 2j, {1: [1]}])
def test_invalid_action_leaves_are_rejected(value):
    with pytest.raises(ValueError):
        action_delta(value, value)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_invalid_tolerances_are_rejected(tolerance):
    with pytest.raises(ValueError):
        within(action_delta([1], [1]), tolerance)


def test_contract_probe_detects_a_same_size_wrong_axis_canary(monkeypatch):
    from boundary_probes import contract
    from boundary_probes.identity import IdentityProbeModel

    class WrongAxisModel(IdentityProbeModel):
        def get_action_batch(self, env_idx_list=None):
            batches = super().get_action_batch(env_idx_list)
            return [[{key: value.reshape(1, -1) for key, value in step.items()}
                     for step in chunk] for chunk in batches]

    monkeypatch.setattr(contract, "IdentityProbeModel", WrongAxisModel)
    report = contract.run_contract_probe()
    assert report["status"] == "fail"
    assert report["checks"]["single_batch_parity"]["status"] == "fail"
