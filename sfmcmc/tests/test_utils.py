import pytest
import numpy as np
import fenics as fe

import jax.numpy as jnp
from scipy.sparse import csr_matrix, rand
from petsc4py.PETSc import Mat

from sfmcmc.samplers import PoissonUnitTheta
from sfmcmc.utils import (build_observation_operator, dolfin_to_csr,
                          sq_exp_covariance, cartesian, kron_matvec,
                          SquareExpKronecker, sparse_to_jax)


@pytest.fixture
def pois():
    return PoissonUnitTheta(32)


def test_sparse_to_jax():
    n = 100
    A = rand(n, n, format="csr")
    x = np.random.normal(size=(n, ))

    A_jax = sparse_to_jax(A)
    assert "rows" in A_jax
    assert "cols" in A_jax
    assert "data" in A_jax


def test_build_observation_operator():
    # unit interval, P1 elements
    mesh = fe.UnitIntervalMesh(128)
    V = fe.FunctionSpace(mesh, "P", 1)

    x_obs = np.linspace(0, 1, 30).reshape((30, 1))
    H = build_observation_operator(x_obs, V)

    assert H.shape == (30, 129)
    assert isinstance(H, csr_matrix)
    np.testing.assert_allclose(np.sum(H, axis=1), 1.)

    H_petsc = build_observation_operator(x_obs, V, out="petsc")
    assert isinstance(H_petsc, Mat)
    with pytest.raises(ValueError):
        build_observation_operator(x_obs, V, out="fenics")

    # check application of operator
    step = 4
    u = fe.Function(V)
    u_np = np.copy(u.vector()[:])
    u_np[::step] = 2.
    H = build_observation_operator(V.tabulate_dof_coordinates()[::step], V)
    np.testing.assert_allclose(H @ u_np, 2.)

    # check that we raise error if outside the mesh
    x_obs = np.linspace(1, 2, 20).reshape((20, 1))
    with pytest.raises(IndexError):
        H = build_observation_operator(x_obs, V)


def test_dolfin_to_csr():
    # unit interval, P1 elements
    mesh = fe.UnitIntervalMesh(32)
    V = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # check dolfin matrix
    form = u * v * fe.dx
    M = fe.assemble(form)
    M_csr = dolfin_to_csr(M)
    assert M_csr.shape == (33, 33)
    assert isinstance(M_csr, csr_matrix)

    # check PETScMatrix
    M = fe.PETScMatrix()
    fe.assemble(form, tensor=M)
    M_csr = dolfin_to_csr(M)
    assert M_csr.shape == (33, 33)
    assert isinstance(M_csr, csr_matrix)


def test_kron_matvec():
    n = 32
    A = np.random.normal(size=(n, n))
    B = np.random.uniform(size=(n, n))

    x = np.ones(n**2)
    A_kron_B = np.kron(A, B)
    np.testing.assert_allclose(kron_matvec(A, B, x), A_kron_B @ x)


def test_square_exp_kronecker(pois):
    scale, ell = 1., 0.1
    grid = pois.mesh.coordinates()
    gp = SquareExpKronecker(grid, scale, ell)

    assert gp.P is None

    np.testing.assert_allclose(cartesian(gp.x1, gp.x2), grid)

    gp.set_permutation(pois.V)
    np.testing.assert_allclose(gp.P.T @ pois.V.tabulate_dof_coordinates(),
                               gp.grid)
    PPT = gp.P @ gp.P.T
    np.testing.assert_allclose(PPT.todense(), np.eye(gp.P.shape[0]))

    K_true = sq_exp_covariance(grid, scale, ell)
    K_kron = np.kron(gp.K_x1, gp.K_x2)
    np.testing.assert_allclose(K_kron, K_true)

    samples = gp.sample(1)
    assert samples.shape == (1089, )

    samples = gp.sample(5)
    assert samples.shape == (1089, 5)
