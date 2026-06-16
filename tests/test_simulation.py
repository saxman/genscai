"""Tests for the modular compartmental disease-modeling engine (genscai.simulation).

The engine integrates a single coupled SEIR(+hospitalization) system per step; the agent's
only structural choice is which mechanism modules to include. These tests pin down that the
physics is consistent (conservation, threshold behavior), that disease burden and
healthcare-capacity effects behave correctly, and that the rigorous-analysis methods
(calibration, optimization, ensemble uncertainty, sensitivity) do what they claim.
"""

import numpy as np
import pytest

from genscai.simulation import Simulation


def make_baseline(**overrides):
    """A well-mixed SEIR scenario above the epidemic threshold, with sensible defaults."""
    params = dict(
        mixing="mass_action",
        population=1_000_000,
        initial_infected=100,
        r0=2.5,
        incubation_days=5.0,
        infectious_days=7.0,
    )
    params.update(overrides)
    sim = Simulation()
    sim.assemble(**params)
    return sim


# -- conservation and threshold behavior ------------------------------------------------


def test_compartments_conserved():
    sim = make_baseline()
    for _ in range(10):
        sim.advance(20)
        state = sim.state()
        living_and_dead = (
            state["susceptible"]
            + state["exposed"]
            + state["infectious"]
            + state["recovered"]
            + state["hospitalized"]
            + state["deaths"]
        )
        assert living_and_dead == pytest.approx(1_000_000, rel=1e-4)


def test_subthreshold_outbreak_fizzles():
    sim = make_baseline(r0=0.5)
    sim.advance(365)
    assert sim.metrics()["attack_rate"] < 0.05


def test_superthreshold_epidemic_peaks():
    sim = make_baseline(r0=2.5)
    sim.advance(365)
    metrics = sim.metrics()
    assert metrics["peak_day"] > 0
    assert metrics["peak_infectious"] > 100
    assert metrics["attack_rate"] > 0.5


def test_no_transmission_when_r0_zero():
    sim = make_baseline(r0=0.0)
    sim.advance(180)
    assert sim.metrics()["cumulative_infections"] == pytest.approx(0.0, abs=1.0)


# -- interventions ----------------------------------------------------------------------


def test_contact_reduction_lowers_peak():
    baseline_peak = make_baseline().advance(365).metrics()["peak_infectious"]

    sim = make_baseline()
    sim.apply_intervention("contact_reduction", magnitude=0.5, start_day=0)
    sim.advance(365)

    assert sim.metrics()["peak_infectious"] < baseline_peak


def test_vaccination_lowers_attack_rate():
    baseline_attack = make_baseline().advance(365).metrics()["attack_rate"]

    sim = make_baseline()
    sim.apply_intervention("vaccination", magnitude=0.02, start_day=0)
    sim.advance(365)

    assert sim.metrics()["attack_rate"] < baseline_attack


# -- mechanism modules ------------------------------------------------------------------


def test_metapopulation_spread_requires_migration():
    isolated = make_baseline(mixing="metapopulation", num_patches=3, migration_rate=0.0)
    isolated.advance(365)
    other_patches_isolated = isolated.infectious_by_patch()[1:]

    connected = make_baseline(mixing="metapopulation", num_patches=3, migration_rate=0.05)
    connected.advance(365)
    cumulative_other_patches = connected.cumulative_infections_by_patch()[1:]

    assert np.allclose(other_patches_isolated, 0.0, atol=1.0)
    assert np.all(cumulative_other_patches > 1.0)


def test_network_density_increases_peak():
    sparse = make_baseline(mixing="network", mean_contacts=8.0)
    sparse.advance(365)

    dense = make_baseline(mixing="network", mean_contacts=20.0)
    dense.advance(365)

    assert dense.metrics()["peak_infectious"] > sparse.metrics()["peak_infectious"]


def test_vector_transmission_requires_biting():
    no_bite = make_baseline(include_vector=True, biting_rate=0.0)
    no_bite.advance(365)

    biting = make_baseline(include_vector=True, biting_rate=0.5)
    biting.advance(365)

    assert no_bite.metrics()["attack_rate"] < 0.01
    assert biting.metrics()["attack_rate"] > 0.1


# -- checkpoint / rewind / reset --------------------------------------------------------


def test_checkpoint_and_rewind_restore_state():
    sim = make_baseline()
    sim.advance(30)
    checkpointed = sim.state()
    sim.checkpoint("day30")

    sim.advance(60)
    assert sim.day == pytest.approx(90)

    sim.rewind("day30")
    assert sim.day == pytest.approx(30)
    restored = sim.state()
    for key, value in checkpointed.items():
        assert restored[key] == pytest.approx(value)


def test_reset_returns_to_day_zero():
    sim = make_baseline()
    sim.advance(120)
    sim.reset()

    assert sim.day == pytest.approx(0)
    assert sim.state()["infectious"] == pytest.approx(100)


def test_reconfigure_changes_parameter_and_resets():
    sim = make_baseline(r0=2.5)
    sim.advance(50)
    sim.reconfigure(r0=0.0)

    assert sim.day == pytest.approx(0)
    sim.advance(180)
    assert sim.metrics()["cumulative_infections"] == pytest.approx(0.0, abs=1.0)


def test_rewind_after_intervention_allows_retry():
    sim = make_baseline()
    sim.apply_intervention("contact_reduction", magnitude=0.3, start_day=0)
    sim.advance(60)
    sim.reset()

    assert sim.list_interventions() == []
    assert sim.day == pytest.approx(0)


# -- disease burden and healthcare capacity ---------------------------------------------


def test_burden_accrues_in_epidemic():
    sim = make_baseline(hospitalized_fraction=0.05, hospital_fatality_fraction=0.2)
    sim.advance(365)
    metrics = sim.metrics()
    assert metrics["peak_hospitalized"] > 0
    assert metrics["cumulative_deaths"] > 0


def test_no_burden_without_hospitalization():
    sim = make_baseline(hospitalized_fraction=0.0)
    sim.advance(365)
    metrics = sim.metrics()
    assert metrics["peak_hospitalized"] == pytest.approx(0.0, abs=1.0)
    assert metrics["cumulative_deaths"] == pytest.approx(0.0, abs=1.0)


def test_exceeding_capacity_raises_deaths():
    ample = make_baseline(hospitalized_fraction=0.08, hospital_capacity=1_000_000, crisis_mortality_multiplier=3.0)
    ample.advance(365)

    scarce = make_baseline(hospitalized_fraction=0.08, hospital_capacity=2_000, crisis_mortality_multiplier=3.0)
    scarce.advance(365)

    assert scarce.metrics()["days_over_capacity"] > 0
    assert scarce.metrics()["cumulative_deaths"] > ample.metrics()["cumulative_deaths"]


# -- calibration ------------------------------------------------------------------------


def test_calibrate_recovers_true_r0():
    truth = make_baseline(r0=2.3)
    times, cumulative = truth.cumulative_infections_series_to(40)
    observed_days = [10.0, 20.0, 30.0, 40.0]
    observed = list(np.interp(observed_days, times, cumulative))

    fit = make_baseline(r0=1.0)  # deliberately wrong starting guess
    result = fit.calibrate_r0(observed_days, observed)

    assert result["r0"] == pytest.approx(2.3, rel=0.1)


# -- optimization -----------------------------------------------------------------------


def test_optimize_intervention_finds_minimal_magnitude():
    sim = make_baseline()
    threshold = 20_000
    result = sim.optimize_intervention(
        kind="contact_reduction", metric="peak_infectious", threshold=threshold, days=365, low=0.0, high=0.95
    )

    assert result["met"]
    assert result["value"] <= threshold * 1.02
    # A meaningfully weaker intervention should fail to meet the threshold (near-minimal).
    weaker = make_baseline()
    weaker.apply_intervention("contact_reduction", magnitude=max(result["magnitude"] - 0.1, 0.0), start_day=0)
    weaker.advance(365)
    assert weaker.metrics()["peak_infectious"] > threshold


# -- ensemble uncertainty and sensitivity -----------------------------------------------


def test_ensemble_reports_intervals_and_is_reproducible():
    sim = make_baseline()
    first = sim.run_ensemble(days=365, n_samples=24, seed=7)
    second = sim.run_ensemble(days=365, n_samples=24, seed=7)
    third = sim.run_ensemble(days=365, n_samples=24, seed=8)

    peak = first["summary"]["peak_infectious"]
    assert peak["p05"] <= peak["median"] <= peak["p95"]
    assert first["summary"]["peak_infectious"]["median"] == second["summary"]["peak_infectious"]["median"]
    assert first["summary"]["peak_infectious"]["median"] != third["summary"]["peak_infectious"]["median"]


def test_sensitivity_ranks_transmission_as_top_driver():
    sim = make_baseline()
    drivers = sim.sensitivity_analysis(days=365, target="peak_infectious", n_samples=64, seed=3)

    ranked = {d["parameter"]: d["correlation"] for d in drivers}
    assert "r0" in ranked
    # r0 should be a strong positive driver of peak prevalence and the top-ranked one.
    assert ranked["r0"] > 0.3
    assert drivers[0]["parameter"] == "r0"
