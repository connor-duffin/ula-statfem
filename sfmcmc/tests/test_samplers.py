import pytest

import numpy as np
import fenics as fe
import jax.numpy as jnp

from pyDOE import lhs

from scipy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from sksparse.cholmod import Factor as CholFactor

from sfmcmc.samplers import (PoissonBase, PoissonUnitTheta,
                             PoissonUnitThetaPosterior, NonlinearPoisson1D)


def grad_approx(u, d, fn, grad_fn, eps=1e-6):
    """ Rel. gradient error from a finite-diff approximation. """
    grad_log_target = grad_fn(u)
    grad = grad_log_target @ d

    u_fwd, u_bwd = u + eps * d, u - eps * d
    fn_fwd = fn(u_fwd)
    fn_bwd = fn(u_bwd)

    grad_fd = (fn_fwd - fn_bwd) / (2 * eps)

    return (np.abs(grad_fd - grad) / np.abs(grad))


@pytest.fixture
def prior_pois():
    pois = PoissonUnitTheta(32)
    pois.setup_G(scale=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")
    return pois


@pytest.fixture
def post_pois():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(scale=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")

    x_obs = lhs(2, 50)
    pois.setup_dgp(x_obs, n_obs=10, sigma=1e-3, scale_factor=1.)
    pois.generate_data()
    pois.setup_jax()
    return pois


def test_init_base():
    pois = PoissonBase(32)
    pois.assemble_A()

    u = np.ones(shape=(pois.n_dofs, ))
    u[pois.bc_dofs] = 0.

    rtol = 1e-6
    np.random.seed(27)
    for i in range(5):
        d = 0.01 * np.random.normal(size=(pois.n_dofs, ))
        rel_error = grad_approx(u, d, pois.residual_form,
                                pois.grad_residual_form, eps=1e-4)
        print(rel_error)
        assert rel_error <= rtol

    pois.gradient_step(u, 1e-4)


def test_init_prior():
    pois = PoissonUnitTheta(32)
    # check sparse G calculations
    pois.setup_G(scale=0.05)
    assert pois.G_dense is None

    assert pois.n_dofs == 1089
    assert pois.x_dofs.shape == (1089, 2)
    assert pois.G.shape == (1089, 1089)

    # check dense G calculations (very poorly conditioned)
    pois.setup_G(0.05, True, 0.2)
    b = 1e-6 * np.ones(shape=(pois.n_dofs, ))
    x = solve(pois.G_dense, b)
    rel_diff = np.abs(x - pois.G_inv @ b) / np.linalg.norm(x)
    assert np.all(rel_diff < 1e-6)

    # check dense G settings
    from scipy.sparse import isspmatrix_dia
    assert isspmatrix_dia(pois.G_inv_sparse)
    assert isspmatrix_dia(pois.G_sqrt_sparse)
    assert pois.G_dense is not None

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


def test_setup_jax(post_pois):
    post_pois.setup_jax()

    assert isinstance(post_pois.H_jax, dict)
    assert post_pois.use_jax
    assert callable(post_pois.grad_log_likelihood_jax)


def test_prior_mala(prior_pois):
    prior_pois.setup_pc("lu")

    _ = prior_pois.mala_step(eta=1e-8)
    A_prev = prior_pois.A.copy()
    accepted = prior_pois.mala_step(eta=1e-8, fixed_theta=True)
    A = prior_pois.A.copy()

    assert type(accepted) is bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())

    _ = prior_pois.pmala_step(eta=1e-8)
    A_prev = prior_pois.A.copy()
    accepted = prior_pois.pmala_step(eta=1e-8, fixed_theta=True)
    A = prior_pois.A.copy()

    assert type(accepted) is bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())


def test_prior_gradients(prior_pois):
    prior_pois.sample_theta()
    prior_pois.assemble_A()

    u = np.ones(shape=(prior_pois.n_dofs, ))
    u[prior_pois.bc_dofs] = 0.

    rtol = 1e-6
    np.random.seed(27)
    for i in range(5):
        d = 0.01 * np.random.normal(size=(prior_pois.n_dofs, ))
        rel_error = grad_approx(u, d, prior_pois.log_target,
                                lambda x: -prior_pois.grad_phi(x))
        print(rel_error)
        assert rel_error <= rtol


def test_init_posterior():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(scale=0.05)

    assert pois.n_dofs == 1089
    assert pois.x_dofs.shape == (1089, 2)
    assert pois.G.shape == (1089, 1089)

    pois.setup_theta(0.1, 0.2, method="kronecker")
    assert pois.theta_method == "kronecker"


def test_posterior_data():
    pois = PoissonUnitThetaPosterior(32)
    pois.setup_G(scale=0.05)
    pois.setup_theta(0.1, 0.2, method="kronecker")

    x_obs = lhs(2, 50)
    pois.setup_dgp(x_obs, n_obs=10, sigma=1e-3, scale_factor=1.)

    assert pois.R.shape == (50, 50)
    assert pois.y.shape == (50, 10)  # 10 obs of 50 points

    assert pois.M.shape == (1089, 1089)
    assert isinstance(pois.M, csc_matrix)
    assert isinstance(pois.M_chol, CholFactor)  # numeric PC factor
    assert isinstance(pois.factor, CholFactor)  # symbolic factor

    pois.generate_data()
    assert pois.y_mean.shape == (pois.n_y, )

    y_mean = np.sum(pois.y, axis=1) / pois.n_obs
    np.testing.assert_allclose(y_mean, pois.y_mean)

    pois.setup_dgp(x_obs,
                   n_obs=10,
                   sigma=1e-3,
                   scale_factor=1.,
                   obs_function=np.exp)
    pois.generate_data()


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


def test_log_likelihood_jax(post_pois):
    def grad_approx_jax(u, d, eps=1e-6):
        """ Rel. gradient error from a finite-diff approximation. """
        grad_ll = post_pois.grad_log_likelihood_jax(post_pois.y,
                                                    post_pois.H_jax, u,
                                                    post_pois.u_obs)
        grad = grad_ll @ d

        u_fwd, u_bwd = u + eps * d, u - eps * d
        lp_fwd = post_pois.log_likelihood_jax(post_pois.y, post_pois.H_jax,
                                              u_fwd, post_pois.u_obs)
        lp_bwd = post_pois.log_likelihood_jax(post_pois.y, post_pois.H_jax,
                                              u_bwd, post_pois.u_obs)
        grad_fd = (lp_fwd - lp_bwd) / (2 * eps)

        return (np.abs(grad_fd - grad) / np.abs(grad))

    # check operation with identity observation function
    u = np.random.normal(size=(post_pois.n_dofs, ))
    d = np.ones((post_pois.n_dofs, ))
    post_pois.setup_jax()
    ll = post_pois.log_likelihood_jax(post_pois.y, post_pois.H_jax, u,
                                      post_pois.u_obs)
    assert ll.dtype == jnp.float64

    rtol = 1e-6
    for i in range(5):
        u = np.random.normal(size=(post_pois.n_dofs, ))
        rdiff = grad_approx_jax(u, d)
        assert rdiff <= rtol

    # check operation with nonlinear observation function
    x_obs = lhs(2, 50)
    post_pois.setup_dgp(x_obs,
                        n_obs=10,
                        sigma=1e-3,
                        scale_factor=1.,
                        obs_function=jnp.sin)
    post_pois.generate_data()
    ll = post_pois.log_likelihood_jax(post_pois.y, post_pois.H_jax, u,
                                      post_pois.u_obs)

    assert ll.dtype == jnp.float64

    for i in range(5):
        u = np.random.normal(size=(post_pois.n_dofs, ))
        rdiff = grad_approx_jax(u, d)
        assert rdiff <= rtol


def test_posterior_gradients(post_pois):
    u = np.ones(shape=(post_pois.n_dofs, ))
    u[post_pois.bc_dofs] = 0.

    rtol = 1e-6
    np.random.seed(27)
    for i in range(5):
        d = 0.01 * np.random.normal(size=(post_pois.n_dofs, ))
        rel_error = grad_approx(u, d, post_pois.log_target,
                                lambda x: -post_pois.grad_phi(x), 1e-4)
        print(rel_error)
        assert rel_error <= rtol


def test_posterior_mala(post_pois):
    _ = post_pois.mala_step(eta=1e-8)
    A_prev = post_pois.A.copy()
    accepted = post_pois.mala_step(eta=1e-8, fixed_theta=True)
    A = post_pois.A.copy()

    assert type(accepted) is bool
    np.testing.assert_allclose(A.todense(), A_prev.todense())


def test_posterior_pcn(post_pois):
    assert not hasattr(post_pois, "A_factor")

    post_pois.pcn_step()

    assert hasattr(post_pois, "A_factor")
    assert callable(post_pois.A_factor.solve_A)

    post_pois.pcn_step(fixed_theta=True)


def test_init_nonlinear():
    pois = NonlinearPoisson1D(32)

    assert pois.n_dofs == 33
    assert pois.x_dofs.shape[0] == pois.n_dofs

    np.testing.assert_allclose(pois.f.vector()[:],
                               8 * np.sin(3 * np.pi * pois.x_dofs[:, 0]**2))


def test_setup_pc_nonlinear():
    pois = NonlinearPoisson1D(32)
    pois.setup_G(0.05)
    pois.setup_pc()

    # check that the setup is OK
    assert isinstance(pois.M_inv, csc_matrix)
    assert isinstance(pois.M_inv_chol, CholFactor)
    assert callable(pois.M_inv_chol.solve_A)


def test_trace_computation_1d():
    pois = NonlinearPoisson1D(32)
    u_np = 1 + 0.2 * np.random.normal(size=(pois.n_dofs, ))

    F, J = pois.assemble_system(u_np)
    J_lu = splu(J.tocsc())
    trace_fast = pois.compute_trace_derivative(J_lu)

    for i in [10, 15, 20, 25, 30]:
        d = fe.Function(pois.V)
        d.vector()[i] = 1.
        H = fe.assemble(fe.derivative(pois.J, pois.u, d)).array()
        trace_true = np.trace(J_lu.solve(H))

        np.testing.assert_almost_equal(trace_fast[i], trace_true)


def test_target_gradients_1d():
    pois = NonlinearPoisson1D(64)
    pois.setup_G(0.05)

    eps = 1e-4
    rtol = 1e-6
    u = np.ones_like(pois.u_curr)
    d = 0.01 * np.random.normal(size=(pois.n_dofs, ))

    u[pois.bc_dofs] = 0.
    d[pois.bc_dofs] = 0.

    grad = pois.grad_phi(u)
    f_fwd = pois.log_target(u + eps * d)
    f_bwd = pois.log_target(u - eps * d)

    grad_approx = (f_fwd - f_bwd) / (2 * eps)
    grad_true = np.dot(d, grad)

    rel_error = np.abs(grad_approx - grad_true) / np.abs(grad_true)
    print(rel_error)
    assert rel_error <= rtol


def test_target_hessian_1d():
    pois = NonlinearPoisson1D(32)
    pois.setup_G(0.5)

    H = pois.compute_hessian_map()
    assert H.shape == (33, 33)


@pytest.mark.skip()
def test_target_gradients():
    pois = NonlinearPoisson1D(32)
    pois.setup_G(0.05)

    eps = 1e-4
    rtol = 1e-6
    u = np.ones_like(pois.u_curr)

    for i in range(10):
        d = 0.01 * np.random.normal(size=(pois.n_dofs, ))

        # enforce BC's
        u[pois.bc_dofs] = 0.
        d[pois.bc_dofs] = 0.

        F, J = pois.assemble_system(u)
        grad_f = pois.grad_log_target(F, J)

        F_fwd, _ = pois.assemble_system(u + eps * d)
        f_fwd = pois.log_target(F_fwd)

        F_bwd, _ = pois.assemble_system(u - eps * d)
        f_bwd = pois.log_target(F_bwd)

        grad_approx = (f_fwd - f_bwd) / (2 * eps)
        grad_true = np.dot(d, grad_f)

        rel_error = np.abs(grad_approx - grad_true) / np.abs(grad_true)
        assert rel_error <= rtol
