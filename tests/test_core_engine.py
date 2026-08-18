from dens_city.core.bindings import DensCityEngine


def test_core_engine_initialization():
    cfg = {
        "T": 300.0,
        "mu": -3000.0 * 1.380649e-23,
        "box_x": 20.0,
        "box_y": 20.0,
        "box_z": 20.0,
        "molecule_type": "single",
        "electrostatics_mode": "short_range",
    }
    eng = DensCityEngine(cfg)
    eng.set_pair_potential(0, 0, kind=1, epsilon_lj=1.0e-21, sigma_lj=3.0, rc=8.0)
    assert eng.number == 0
    assert abs(eng.total_energy()) < 1e-12


def test_core_engine_gcmc_steps():
    cfg = {
        "T": 300.0,
        "mu": -1000.0 * 1.380649e-23,
        "box_x": 20.0,
        "box_y": 20.0,
        "box_z": 20.0,
        "molecule_type": "single",
        "prob_insert": 0.5,
        "prob_delete": 0.2,
        "prob_displace": 0.3,
    }
    eng = DensCityEngine(cfg)
    eng.set_pair_potential(0, 0, kind=1, epsilon_lj=1.0e-21, sigma_lj=3.0, rc=8.0)
    eng.run_steps(500)
    assert eng.number > 0


def test_core_engine_long_range_ewald():
    cfg = {
        "T": 300.0,
        "mu": -2000.0 * 1.380649e-23,
        "box_x": 20.0,
        "box_y": 20.0,
        "box_z": 20.0,
        "molecule_type": "abc",
        "electrostatics_mode": "long_range",
        "ewald_alpha": 0.35,
        "ewald_kmax": 4,
        "prob_insert": 0.5,
        "prob_delete": 0.2,
        "prob_displace": 0.3,
    }
    eng = DensCityEngine(cfg)
    eng.set_pair_potential(0, 0, kind=5, epsilon_lj=1.0e-21, sigma_lj=3.0, rc=8.0, q1=-0.382, q2=-0.382)
    eng.run_steps(200)
    assert eng.number >= 0
