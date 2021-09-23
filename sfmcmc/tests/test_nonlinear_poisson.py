import pytest

import numpy as np
import fenics as fe

from scipy.sparse.linalg import splu

from sfmcmc.nonlinear_poisson import NonlinearPoisson1D, NonlinearPoisson


def test_trace_computation():
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
    assert rel_error <= rtol


@pytest.mark.skip()
def test_target_gradients():
    pois = NonlinearPoisson(32)
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
