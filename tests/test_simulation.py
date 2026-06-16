"""Tests for the modular compartmental disease-modeling engine (genscai.simulation).

The engine integrates a single coupled SEIR system per step; the agent's only structural
choice is which mechanism modules to include. These tests pin down that the physics is
consistent (conservation, threshold behavior) and that each module and intervention has the
expected effect.
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


def test_compartments_conserved():
    sim = make_baseline()
    for _ in range(10):
        sim.advance(20)
        state = sim.state()
        total = state["susceptible"] + state["exposed"] + state["infectious"] + state["recovered"]
        assert total == pytest.approx(1_000_000, rel=1e-4)


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


def test_rewind_after_intervention_allows_retry():
    sim = make_baseline()
    sim.apply_intervention("contact_reduction", magnitude=0.3, start_day=0)
    sim.advance(60)
    sim.reset()

    assert sim.list_interventions() == []
    assert sim.day == pytest.approx(0)
