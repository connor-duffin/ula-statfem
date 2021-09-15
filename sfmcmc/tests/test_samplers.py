import pytest

import numpy as np

from pyDOE import lhs

from scipy.sparse import csc_matrix, csr_matrix
from sksparse.cholmod import Factor as CholFactor

from sfmcmc.samplers import (PoissonUnitTheta, PoissonUnitThetaPosterior)


@pytest.fixture
def prior_pois():
    pois = PoissonUnitTheta(32)
    pois.setup_G(sigma=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")
    return pois


@pytest.fixture
def post_pois():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(sigma=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")

    x_obs = lhs(2, 50)
    pois.setup_dgp(x_obs, n_obs=10, sigma=1e-3, scale_factor=1.)
    pois.setup_pc_post()
    pois.generate_data()
    return pois


def test_init_prior():
    pois = PoissonUnitTheta(32)
    pois.setup_G(sigma=0.05)

    assert pois.n_dofs == 1089
    assert pois.x_dofs.shape == (1089, 2)
    assert pois.G.shape == (1089, 1089)

    pois.setup_theta(0.1, 0.2, method="kronecker")
    assert pois.theta_method == "kronecker"


def test_setup_pc(prior_pois):
    prior_pois.setup_pc("lr")
    assert callable(prior_pois.M.matvec)
    assert callable(prior_pois.M.matvec_sqrt)

    z = np.random.normal(size=(prior_pois.n_dofs, ))
    assert prior_pois.M.matvec(z).shape == z.shape
    assert prior_pois.M.matvec_sqrt(z).shape == z.shape

    prior_pois.setup_pc("amg")
    assert callable(prior_pois.M.solve)

    prior_pois.setup_pc("lu")
    assert callable(prior_pois.M.solve_A)

    prior_pois.setup_pc("chol")
    assert callable(prior_pois.M.solve_A)

    prior_pois.setup_pc("ilu")
    assert callable(prior_pois.M.solve)


def test_prior_mala(prior_pois):
    prior_pois.setup_pc("lu")

    _ = prior_pois.mala_step(eta=1e-8)
    A_prev = prior_pois.A.copy()
    accepted = prior_pois.mala_step(eta=1e-8, fixed_theta=True)
    A = prior_pois.A.copy()

    assert type(accepted) == bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())

    _ = prior_pois.pmala_step(eta=1e-8)
    A_prev = prior_pois.A.copy()
    accepted = prior_pois.pmala_step(eta=1e-8, fixed_theta=True)
    A = prior_pois.A.copy()

    assert type(accepted) == bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())


def test_prior_gradients(prior_pois):
    prior_pois.sample_theta()
    prior_pois.assemble_A()

    def grad_approx(u, d, eps=1e-6):
        """ Rel. gradient error from a finite-diff approximation. """
        grad_log_target = -prior_pois.grad_phi(u)
        grad = grad_log_target @ d

        u_fwd, u_bwd = u + eps * d, u - eps * d
        lp_fwd = prior_pois.log_target(u_fwd)
        lp_bwd = prior_pois.log_target(u_bwd)
        grad_fd = (lp_fwd - lp_bwd) / (2 * eps)

        return (np.abs(grad_fd - grad) / np.abs(grad))

    u = np.ones(shape=(prior_pois.n_dofs, ))
    u[prior_pois.bc_dofs] = 0.

    rtol = 1e-6
    np.random.seed(27)
    for i in range(5):
        d = 0.01 * np.random.normal(size=(prior_pois.n_dofs, ))
        rel_error = grad_approx(u, d)
        print(rel_error)
        assert rel_error <= rtol


def test_init_posterior():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(sigma=0.05)

    assert pois.n_dofs == 1089
    assert pois.x_dofs.shape == (1089, 2)
    assert pois.G.shape == (1089, 1089)

    pois.setup_theta(0.1, 0.2, method="kronecker")
    assert pois.theta_method == "kronecker"


def test_setup_pc_posterior(post_pois):
    post_pois.setup_pc_post()

    assert type(post_pois.M) == csc_matrix
    assert type(post_pois.M_chol) == CholFactor


def test_posterior_data():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(sigma=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")

    x_obs = lhs(2, 50)
    pois.setup_dgp(x_obs, n_obs=10, sigma=1e-3, scale_factor=1.)

    assert pois.R.shape == (50, 50)
    assert pois.y.shape == (50, 10)  # 10 obs of 50 points

    pois.setup_pc_post()
    assert pois.M.shape == (1089, 1089)
    assert type(pois.M) == csc_matrix
    assert type(pois.M_chol) == CholFactor  # numeric PC factor
    assert type(pois.factor) == CholFactor  # symbolic factor

    pois.generate_data()
    assert pois.y_mean.shape == (pois.n_y, )

    y_mean = np.sum(pois.y, axis=1) / pois.n_obs
    np.testing.assert_allclose(y_mean, pois.y_mean)


def test_log_likelihood(post_pois):
    def log_likelihood_dumb(u):
        ll = 0
        for i in range(post_pois.n_obs):
            resid = post_pois.y[:, i] - post_pois.H @ u
            ll -= np.dot(resid, post_pois.R_inv @ resid) / 2
        return ll

    for i in range(5):
        u = np.random.normal(size=(post_pois.n_dofs, ))
        u[post_pois.bc_dofs] = 0.

        ll_true = post_pois.log_likelihood(u)
        ll_dumb = log_likelihood_dumb(u)

        print(ll_true - ll_dumb)
        np.testing.assert_approx_equal(ll_true, ll_dumb)


def test_posterior_gradients(post_pois):
    def grad_approx(u, d, eps=1e-6):
        """ Rel. gradient error from a finite-diff approximation. """
        grad_log_target = -post_pois.grad_phi(u)
        grad = grad_log_target @ d

        u_fwd, u_bwd = u + eps * d, u - eps * d
        lp_fwd = post_pois.log_target(u_fwd)
        lp_bwd = post_pois.log_target(u_bwd)
        grad_fd = (lp_fwd - lp_bwd) / (2 * eps)

        return (np.abs(grad_fd - grad) / np.abs(grad))

    u = np.ones(shape=(post_pois.n_dofs, ))
    u[post_pois.bc_dofs] = 0.

    rtol = 1e-6
    np.random.seed(27)
    for i in range(5):
        d = 0.01 * np.random.normal(size=(post_pois.n_dofs, ))
        rel_error = grad_approx(u, d)
        assert rel_error <= rtol


def test_posterior_mala(post_pois):
    _ = post_pois.mala_step(eta=1e-8)
    A_prev = post_pois.A.copy()
    accepted = post_pois.mala_step(eta=1e-8, fixed_theta=True)
    A = post_pois.A.copy()

    assert type(accepted) == bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())


def test_posterior_pcn(post_pois):
    assert not hasattr(post_pois, "A_factor")

    post_pois.pcn_step()

    assert hasattr(post_pois, "A_factor")
    assert callable(post_pois.A_factor.solve_A)

    post_pois.pcn_step(fixed_theta=True)
