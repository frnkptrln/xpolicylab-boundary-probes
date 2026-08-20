from boundary_probes.contract import run_contract_probe
from boundary_probes.reset_leakage import (
    StatefulCanaryModel,
    probe_factory,
    run_reset_probe,
)


def test_contract_probe_passes():
    result = run_contract_probe()
    assert result["status"] == "pass"
    assert result["checks"]["single_batch_parity"]["status"] == "pass"
    assert result["checks"]["batch_ordering"]["status"] == "pass"
    assert result["checks"]["nontrivial_identity"]["distinct_signatures"] == 10


def test_reset_probe_passes_only_when_known_bad_canary_is_detected():
    result = run_reset_probe()
    assert result["status"] == "pass"
    assert result["checks"]["identity_probe"]["status"] == "pass"
    assert result["checks"]["correct_reset"]["status"] == "pass"
    assert result["checks"]["known_bad_reset"]["status"] == "pass"
    assert result["checks"]["known_bad_reset"]["detector_observed"] == "fail"


def test_reset_protocol_flags_leaky_policy():
    result = probe_factory(lambda: StatefulCanaryModel(clear_on_reset=False))
    assert result["status"] == "fail"
    assert result["max_abs"] > 0
