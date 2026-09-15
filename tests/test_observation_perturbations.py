from boundary_probes.cli import _record
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
