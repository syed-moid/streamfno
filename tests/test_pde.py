"""Fokker-Planck solver tests: closed-form stationary validation, mass
conservation, regulator behavior, and refinement stability."""

import numpy as np

from streamfno.pde.solver import FPResult, solve_fp, stationary_exponential


def _uniform_rho0(m_cells, n_classes=1):
    return np.ones((n_classes, m_cells))


def _stationary_cell_averages(m_cells, b, a):
    """Cell averages of the closed-form stationary density via its CDF
    (point sampling misrepresents boundary layers thinner than a cell)."""
    edges = np.arange(m_cells + 1) / m_cells
    r = 2.0 * b / a
    cdf = np.expm1(r * edges) / np.expm1(r)
    return np.diff(cdf) * m_cells


def test_stationary_matches_closed_form():
    """Constant b < 0, constant a, reflecting walls: the solver's long-time
    solution must match rho ~ exp(2 b x / a) to discretization accuracy."""
    b, a = -0.3, 0.04
    m_cells = 200
    res = solve_fp(_uniform_rho0(m_cells), lambda x, m: b * np.ones_like(x), a,
                   t_end=30.0, dt=2e-3)
    x = res.x_centers
    exact = stationary_exponential(x, b, a)
    err = np.abs(res.rho[-1, 0] - exact).sum() / m_cells  # L1 error
    assert err < 2e-3, err
    # Chang-Cooper should be near-exact on the exponential profile
    assert np.all(res.rho[-1, 0] > 0.0)


def test_stationary_high_peclet_drain():
    """Strong drain with tiny diffusion (cell Peclet |b| h / D ~ 9, the
    real-telemetry regime a ~ 1e-3): the solver must stay positive,
    conserve mass, and still match the closed-form exponential.
    Regression test for the Chang-Cooper weight orientation -- with the
    weight on the wrong cell the scheme is downwind here and diverges."""
    b, a = -0.3, 1e-3
    m_cells = 128
    res = solve_fp(_uniform_rho0(m_cells), lambda x, m: b * np.ones_like(x), a,
                   t_end=16.0, dt=2e-3)
    h = 1.0 / m_cells
    assert np.all(np.isfinite(res.rho))
    assert np.all(res.rho >= -1e-12)
    np.testing.assert_allclose(res.rho[:, 0, :].sum(axis=1) * h, 1.0,
                               atol=1e-8)
    exact = _stationary_cell_averages(m_cells, b, a)
    # relative L1 on the wall boundary layer where the mass lives
    err = np.abs(res.rho[-1, 0] - exact).sum() / exact.sum()
    assert err < 5e-2, err


def test_stationary_high_peclet_build():
    """Mirror case toward the regulated wall: strong build, tiny
    diffusion (the e07 forecast regime, b = +0.12, a ~ 7e-4)."""
    b, a = 0.12, 7.4e-4
    m_cells = 128
    res = solve_fp(_uniform_rho0(m_cells), lambda x, m: b * np.ones_like(x), a,
                   t_end=60.0, dt=2e-3)
    assert np.all(np.isfinite(res.rho))
    assert np.all(res.rho >= -1e-12)
    exact = _stationary_cell_averages(m_cells, b, a)
    err = np.abs(res.rho[-1, 0] - exact).sum() / exact.sum()
    assert err < 5e-2, err


def test_mass_conservation_and_positivity():
    b, a = 0.25, 0.05  # supercritical: pushes mass to x = 1
    m_cells = 150
    rho0 = _uniform_rho0(m_cells)
    res = solve_fp(rho0, lambda x, m: b * np.ones_like(x), a, t_end=8.0, dt=2e-3)
    h = 1.0 / m_cells
    masses = res.rho[:, 0, :].sum(axis=1) * h
    np.testing.assert_allclose(masses, 1.0, atol=1e-10)
    assert np.all(res.rho >= -1e-12)
    # regulator must activate once mass reaches the wall
    assert res.regulator_cum[-1, 0] > 0.0
    assert np.all(np.diff(res.regulator_cum[:, 0]) >= -1e-14)


def test_stationary_regulator_balances_drift():
    """Constant supercritical b > 0: at stationarity the upper-wall regulator
    (local-time) rate (a/2) rho(1) must balance the drift, dK/dt -> b."""
    b, a = 0.25, 0.05
    res = solve_fp(_uniform_rho0(200), lambda x, m: b * np.ones_like(x), a,
                   t_end=20.0, dt=2e-3)
    exact = 0.5 * a * stationary_exponential(np.array([1.0]), b, a)[0]
    assert abs(exact - b) < 1e-3  # sanity of the closed form itself
    # rho(1) is approximated by the last cell average (O(h) low in the
    # boundary layer), so allow a few percent
    assert abs(res.regulator_rate[-1, 0] - b) / b < 0.05


def test_subcritical_regulator_stays_negligible():
    """Drift away from the wall and no initial mass near it: the local-time
    regulator at x = 1 must stay negligible."""
    b, a = -0.3, 0.04
    m_cells = 100
    x = (np.arange(m_cells) + 0.5) / m_cells
    rho0 = np.exp(-0.5 * ((x - 0.2) / 0.05) ** 2)
    rho0 /= rho0.sum() / m_cells
    res = solve_fp(rho0[None, :], lambda x_, m: b * np.ones_like(x_), a,
                   t_end=10.0, dt=2e-3)
    # stationary rho(1) ~ 5e-6, so the local-time rate is ~1e-7 per unit time
    assert res.regulator_cum[-1, 0] < 1e-5


def test_transport_limit_advects_profile():
    """a = 0 degenerates to upwind transport: a bump moves with speed b."""
    m_cells = 400
    x = (np.arange(m_cells) + 0.5) / m_cells
    rho0 = np.exp(-0.5 * ((x - 0.6) / 0.05) ** 2)
    rho0 /= rho0.sum() / m_cells
    b = -0.2
    res = solve_fp(rho0, lambda xx, m: b * np.ones_like(xx), 0.0,
                   t_end=1.0, dt=5e-4)
    mean0 = res.mean_lag[0, 0]
    mean1 = res.mean_lag[-1, 0]
    np.testing.assert_allclose(mean1 - mean0, b * 1.0, atol=0.01)


def test_mean_field_coupling_feeds_back():
    """Drift depending on the class mean must alter the trajectory."""
    m_cells = 100
    rho0 = _uniform_rho0(m_cells)

    def coupled(x, m):
        return (0.3 - 0.8 * m) * np.ones_like(x)

    res = solve_fp(rho0, coupled, 0.04, t_end=20.0, dt=2e-3)
    # fixed point of m: b(m*) = 0 => m* ~ 0.375; mean must settle near it
    assert abs(res.mean_lag[-1, 0] - 0.375) < 0.05


def test_refinement_stability():
    """Halving dt and doubling the grid must not change the solution beyond
    plotting accuracy (L1 ~ 1e-2)."""
    b, a = -0.1, 0.04

    def run(m_cells, dt):
        return solve_fp(_uniform_rho0(m_cells), lambda x, m: b * np.ones_like(x),
                        a, t_end=3.0, dt=dt)

    coarse = run(100, 4e-3)
    fine = run(200, 2e-3)
    rho_f = fine.rho[-1, 0].reshape(100, 2).mean(axis=1)  # restrict to coarse grid
    err = np.abs(coarse.rho[-1, 0] - rho_f).sum() / 100
    assert err < 1e-2, err


def test_result_io(tmp_path):
    res = solve_fp(_uniform_rho0(50), lambda x, m: -0.1 * np.ones_like(x), 0.04,
                   t_end=0.5, dt=1e-2)
    p = tmp_path / "fp.npz"
    res.save(p)
    back = FPResult.load(p)
    np.testing.assert_array_equal(res.rho, back.rho)
    np.testing.assert_array_equal(res.times, back.times)
