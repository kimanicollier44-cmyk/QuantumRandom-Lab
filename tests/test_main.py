"""Tests for QuantumRandom-Lab: every number checked against the model."""
import math

import pytest

import main as m


# ---- statevector simulator: Born probabilities are the physics
def test_state_normalised_after_hadamard():
    probs = m.born_probabilities(m.hadamard_all(m.zero_state(3), 3))
    assert abs(sum(probs) - 1.0) < 1e-12


def test_hadamard_uniform_probabilities():
    probs = m.born_probabilities(m.hadamard_all(m.zero_state(3), 3))
    assert all(abs(p - 1.0 / 8.0) < 1e-12 for p in probs)


def test_hadamard_prob_one_half_each_qubit():
    st = m.hadamard_all(m.zero_state(4), 4)
    for q in range(4):
        assert abs(m.prob_one(st, 4, q) - 0.5) < 1e-12


def test_zero_state_probabilities():
    assert m.born_probabilities(m.zero_state(2)) == [1.0, 0.0, 0.0, 0.0]


def test_max_qubits_guard():
    with pytest.raises(ValueError):
        m.zero_state(m.MAX_QUBITS + 1)
    with pytest.raises(ValueError):
        m.zero_state(0)


def test_biased_prob_one_pi_over_6():
    # sin^2(pi/6) = 0.25 exactly - the lab's biased teaching source
    assert abs(m.prob_one(m.biased_qubit_state(m.BIASED_THETA), 1, 0)
               - 0.25) < 1e-12


def test_fair_theta_prob_one_half():
    assert abs(m.prob_one(m.biased_qubit_state(m.FAIR_THETA), 1, 0)
               - 0.5) < 1e-12


def test_biased_prob_monotone_in_theta():
    p = [m.prob_one(m.biased_qubit_state(t), 1, 0) for t in (0.3, 0.6, 0.9)]
    assert p[0] < p[1] < p[2]


def test_sampling_counts_sum_to_shots():
    st = m.hadamard_all(m.zero_state(3), 3)
    counts = m.sample_basis_counts(st, 3, 8000)
    assert sum(counts) == 8000


def test_sampling_same_seed_deterministic():
    st = m.hadamard_all(m.zero_state(3), 3)
    assert (m.sample_basis_counts(st, 3, 500)
            == m.sample_basis_counts(st, 3, 500))


def test_sampling_uniform_roughly_even():
    st = m.hadamard_all(m.zero_state(3), 3)
    counts = m.sample_basis_counts(st, 3, 8000)
    assert all(700 < x < 1300 for x in counts)


def test_sampling_biased_skewed():
    counts = m.sample_basis_counts(m.biased_qubit_state(m.BIASED_THETA),
                                   1, 8192)
    assert abs(counts[1] / 8192.0 - 0.25) < 0.03


# ---- entropy accounting: analytic values, then measured streams
def test_shannon_uniform_byte_is_8():
    assert abs(m.shannon_entropy([1.0 / 256.0] * 256) - 8.0) < 1e-9


def test_min_entropy_uniform_byte_is_8():
    assert abs(m.min_entropy([1.0 / 256.0] * 256) - 8.0) < 1e-9


def test_shannon_fair_bit_is_1():
    assert m.shannon_entropy([0.5, 0.5]) == 1.0


def test_min_entropy_biased_bit_value():
    # -log2(0.75): the cautious entropy of the pi/6-biased bit
    assert abs(m.min_entropy([0.75, 0.25]) - 0.415037) < 1e-6


def test_min_entropy_empty_raises():
    with pytest.raises(ValueError):
        m.min_entropy([])


def test_uniform_stream_entropy_high():
    rep = m.entropy_report(m.hadamard_bits(m.FAIR_BITS))
    assert rep["shannon_per_byte"] > 7.9
    assert rep["min_entropy_per_bit"] > 0.97


def test_biased_stream_entropy_lower():
    fair = m.entropy_report(m.hadamard_bits(m.FAIR_BITS))
    biased = m.entropy_report(m.biased_bits(m.BIASED_BITS))
    assert biased["shannon_per_bit"] < fair["shannon_per_bit"]
    assert biased["min_entropy_per_bit"] < fair["min_entropy_per_bit"]
    assert biased["min_entropy_per_byte"] < fair["min_entropy_per_byte"]
    assert abs(biased["bias"] - 0.25) < 0.02


def test_bits_to_bytes_msb_first():
    assert m.bits_to_bytes([1, 0, 0, 0, 0, 0, 0, 0]) == [128]
    assert m.bits_to_bytes([1] * 8) == [255]
    assert m.bits_to_bytes([1, 0, 1]) == []  # trailing partial byte dropped


# ---- von Neumann extractor: mapping, yield, unbiasedness
def test_von_neumann_pair_mapping():
    assert m.von_neumann_extract([0, 1, 1, 0, 0, 0, 1, 1]) == [0, 1]


def test_von_neumann_constant_stream_empty():
    assert m.von_neumann_extract([0] * 100) == []
    assert m.von_neumann_extract([1] * 100) == []


def test_theoretical_yield_values():
    assert abs(m.theoretical_yield(0.25) - 0.1875) < 1e-12
    assert abs(m.pair_acceptance_rate(0.25) - 0.375) < 1e-12
    assert abs(m.theoretical_yield(0.5) - 0.25) < 1e-12


def test_measured_yield_matches_theory():
    ext = m.extraction_report()
    assert abs(ext["measured_yield_per_input_bit"]
               - ext["theoretical_yield_per_input_bit"]) < 0.02
    assert abs(ext["pair_acceptance_measured"]
               - ext["pair_acceptance_theoretical"]) < 0.04


def test_extracted_output_unbiased():
    ext = m.extraction_report()
    assert ext["output_bits"] < ext["input_bits"]
    assert abs(ext["output_p_one"] - 0.5) < 0.03


# ---- the battery: biased raw fails, extraction fixes it, PRNG labelled
def test_battery_fair_passes():
    assert m.run_battery(m.hadamard_bits(m.FAIR_BITS), quantum=True)["passed"]


def test_battery_biased_raw_fails_frequency_and_chi_square():
    result = m.run_battery(m.biased_bits(m.BIASED_BITS), quantum=True)
    by_name = {t["test"]: t for t in result["tests"]}
    assert not by_name["monobit_frequency"]["passed"]
    assert not by_name["chi_square_bytes"]["passed"]
    assert not result["passed"]


def test_battery_extracted_passes():
    extracted = m.von_neumann_extract(m.biased_bits(m.EXTRACT_SOURCE_BITS))
    assert m.run_battery(extracted, quantum=True)["passed"]


def test_battery_prng_passes_but_labelled_not_quantum():
    result = m.run_battery(m.prng_bits(m.FAIR_BITS), quantum=False)
    assert result["passed"]
    assert result["quantum_source"] is False


def test_sticky_fails_autocorrelation_and_runs():
    sticky = m.sticky_bits(m.FAIR_BITS, stick=0.8)
    assert not m.lag1_autocorrelation(sticky)["passed"]
    assert not m.runs_test(sticky)["passed"]


def test_monobit_threshold_explicit():
    result = m.monobit_frequency_test(m.biased_bits(m.BIASED_BITS))
    assert result["statistic"] > m.MONOBIT_Z_MAX
    assert result["threshold"] == m.MONOBIT_Z_MAX


def test_autocorrelation_iid_biased_near_zero():
    # bias is not correlation: an iid biased stream centres to ~0
    result = m.lag1_autocorrelation(m.biased_bits(m.BIASED_BITS))
    assert result["passed"]


def test_chi_square_needs_enough_bytes():
    with pytest.raises(ValueError):
        m.chi_square_bytes_test([0, 1] * 100)


# ---- classifier: agrees with the battery-distilled model
def test_classifier_scenarios():
    rep = m.source_report()
    assert rep["hadamard_fair_source"]["label"] == "good"
    assert rep["slightly_biased_p47"]["label"] == "marginal"
    assert rep["heavily_biased_p25"]["label"] == "bad"
    assert rep["sticky_source"]["label"] == "bad"
    assert rep["extracted_source"]["label"] == "good"


def test_classifier_matches_model_on_scenarios():
    for row in m.source_report().values():
        assert row["label"] == row["model_label"]


def test_classifier_training_labels_match_model():
    for row in m._CLASSIFIER_TRAIN:
        assert m.model_label(row[0] / 4.0, row[1], row[2] / 10.0) == row[3]


def test_model_label_boundaries():
    assert m.model_label(0.0, 1.0, 0.0) == "good"
    assert m.model_label(0.25, 0.415, 0.0) == "bad"
    assert m.model_label(0.0, 1.0, 0.6) == "bad"
    assert m.model_label(0.03, 0.91, 0.0) == "marginal"


def test_classifier_invalid_features_raise():
    with pytest.raises(ValueError):
        m.classify_features(0.9, 1.0, 0.0)
    with pytest.raises(ValueError):
        m.classify_features(0.0, 1.5, 0.0)


# ---- gates + seats
def test_hold_gates_all_present():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_claim_gates_hold():
    for action in ("certified_qrng_claim", "cryptographic_key_generation",
                   "password_or_secret_generation", "gambling_lottery_claim",
                   "randomness_beacon_operation", "live_hardware_control",
                   "seeded_output_as_quantum_claim"):
        assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_never_override():
    b = m.brains_report()
    assert b["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert b["decider"]["can_override_hold"] is False
    assert b["advisor"]["can_override_hold"] is False
    assert b["flow"]["rendered"] is False and b["flow"]["live"] == 0


def test_demo_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


def test_demo_deterministic():
    import io
    import contextlib
    out1, out2 = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out1):
        m.demo()
    with contextlib.redirect_stdout(out2):
        m.demo()
    assert out1.getvalue() == out2.getvalue()
