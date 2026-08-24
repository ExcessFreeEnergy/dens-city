"""
Unit and integration tests for CDFTBGWorker and interactive UI components in dens_city.ui.
Validates non-blocking ZeroMQ streaming, state machine transitions, telemetry calculation,
and MCMC relaxation coordinate updates.
"""

import time

import numpy as np

from dens_city.ui.worker import (
    CDFTBGWorker,
    compute_end_to_end_distance,
    compute_radius_of_gyration,
    count_steric_clashes,
)
from dens_city.utils.materials import MaterialLoader


def test_geometric_metrics_calculation():
    """Validates Rg, Ree, and steric clash counting on known geometric configurations."""
    # 1. Collinear 3-atom test
    coords = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    rg = compute_radius_of_gyration(coords)
    assert rg > 0.0
    ree = compute_end_to_end_distance(coords)
    assert abs(ree - 4.0) < 1e-5

    # 2. Steric clash detection
    sigmas = [3.4, 3.4, 3.4]
    bonds = [(0, 1, "1"), (1, 2, "1")]
    # No non-bonded clashes in normal bonded chain
    clashes = count_steric_clashes(coords, sigmas, bonds)
    assert clashes == 0

    # 3. Severe clash between atom 0 and atom 2
    clash_coords = np.array([[0.0, 0.0, 0.1], [0.0, 1.5, 0.0], [0.0, 0.0, 0.2]])
    clashes_severe = count_steric_clashes(clash_coords, sigmas, bonds)
    assert clashes_severe == 1


def test_cdft_bg_worker_initialization_and_single_step():
    """Validates that CDFTBGWorker initializes cDFT and executes single step with ZeroMQ."""
    water = MaterialLoader.load_material("water")
    worker = CDFTBGWorker(
        material=water,
        n_grid=64,
        cdft_steps=10,
        bg_mcmc_steps=5,
        zmq_endpoint="inproc://test_worker_step",
    )

    try:
        t0 = worker.poll_telemetry()
        assert t0.state == "WAITING_CDFT"
        assert t0.cdft_progress == 0.0
        assert len(t0.rho_z) == 64

        # Execute 1 cDFT step
        worker.step_cdft()
        time.sleep(0.05)
        t1 = worker.poll_telemetry()

        assert t1.cdft_step == 1
        assert t1.cdft_progress > 0.0
        assert len(t1.rho_z) == 64
        assert t1.wall_pressure_bar != 0.0
    finally:
        worker.close()


def test_cdft_bg_worker_full_pipeline_streaming():
    """Validates asynchronous solving of cDFT followed by BG MCMC relaxation on Argon."""
    argon = MaterialLoader.load_material("argon")
    worker = CDFTBGWorker(
        material=argon,
        n_grid=32,
        cdft_steps=5,
        bg_mcmc_steps=5,
        zmq_endpoint="inproc://test_worker_pipeline",
    )

    try:
        # 1. Start asynchronous cDFT solve
        worker.solve_cdft()
        assert worker.is_running is True

        # Wait for cDFT to converge
        for _ in range(120):
            t = worker.poll_telemetry()
            if t.state == "CDFT_CONVERGED" and not worker.is_running:
                break
            time.sleep(0.05)

        t_cdft = worker.poll_telemetry()
        assert t_cdft.state == "CDFT_CONVERGED"
        assert t_cdft.cdft_progress == 1.0

        # 2. Start asynchronous BG solve
        worker.solve_bg()
        time.sleep(0.05)
        assert worker.is_running is True

        # Wait for BG to complete
        for _ in range(120):
            t = worker.poll_telemetry()
            if t.state == "COMPLETE" and not worker.is_running:
                break
            time.sleep(0.05)

        t_complete = worker.poll_telemetry()
        assert t_complete.state == "COMPLETE"
        assert t_complete.bg_progress == 1.0
        assert t_complete.coating_viability != "PENDING"
    finally:
        worker.close()


def test_cdft_bg_worker_cancellation():
    """Validates that worker cancellation stops execution gracefully without errors."""
    water = MaterialLoader.load_material("water")
    worker = CDFTBGWorker(
        material=water,
        n_grid=64,
        cdft_steps=200,
        zmq_endpoint="inproc://test_worker_cancel",
    )

    try:
        worker.solve_cdft()
        time.sleep(0.05)
        assert worker.is_running is True

        worker.cancel()
        time.sleep(0.05)
        assert worker.is_running is False
        t = worker.poll_telemetry()
        assert t.state == "WAITING_CDFT"
    finally:
        worker.close()
