import logging

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe

from petsc4py.PETSc import Mat
from scipy.linalg import cholesky
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)
fe.set_log_level(40)


def sq_exp_covariance(grid, scale, ell, nugget=1e-10):
    """Square-exponential covariance function. """
    dist = pdist(grid, metric="euclidean")
    dist = squareform(dist)

    K = scale**2 * np.exp(-dist**2 / (2 * ell**2))
    K[np.diag_indices_from(K)] += nugget
    return K


def dolfin_to_csr(A):
    """Convert assembled matrix to scipy CSR. """
    if type(A) != Mat:
        mat = fe.as_backend_type(A).mat()
    else:
        mat = A
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr


def kron_matvec(A, B, x):
    """Compute the matvec (A kron B) x. """
    # (A o B) x = vec(B mat(x) A.T)
    n = A.shape[0]
    x_mat = x.reshape(n, n).T
    out_mat = B @ x_mat @ A.T
    return out_mat.T.flatten()


def cartesian(x, y):
    """Cartesian product of x and y.

    Returns [[x0, y0], [x0, y1], ..., [xN, yN]].
    """
    nx, ny = len(x), len(y)
    n_prod = nx * ny
    nx_tile = n_prod // ny
    ny_tile = n_prod // nx
    out = np.zeros((n_prod, 2))
    out[:, 0] = np.tile(y, ny_tile)
    out[:, 1] = np.repeat(x, nx_tile)
    return out


def build_observation_operator(x_obs, V, sub=0, out="scipy"):
    """
    Build interpolation matrix from `x_obs` on function space V. This
    assumes that the observations are from the first sub-function of V.

    From the fenics forums.
    """
    nx, dim = x_obs.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    mesh_cells = mesh.cells()
    bbt = mesh.bounding_box_tree()

    # dofs from first subspace
    if V.num_sub_spaces() > 1:
        dolfin_element = V.sub(sub).dolfin_element()
        dofmap = V.sub(sub).dofmap()
    else:
        dolfin_element = V.dolfin_element()
        dofmap = V.dofmap()

    sdim = dolfin_element.space_dimension()

    v = np.zeros(sdim)
    rows = np.zeros(nx * sdim, dtype='int')
    cols = np.zeros(nx * sdim, dtype='int')
    vals = np.zeros(nx * sdim)

    # loop over all interpolation points
    for k in range(nx):
        x = x_obs[k, :]
        if dim == 1:
            p = fe.Point(x[0])
        elif dim == 2:
            p = fe.Point(x[0], x[1])
        elif dim == 3:
            p = fe.Point(x[0], x[1], x[2])
        else:
            logger.error("no support for higher dims")
            raise ValueError

        # find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)

        # vertex coordinates for the cell
        xvert = coords[mesh_cells[cell_id, :], :]

        # evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x, xvert, cell_id)

        v[v < 1e-10] = 0.  # assume zero contribution below 1e-10

        # set the sparse metadata
        j = np.arange(sdim * k, sdim * (k + 1))
        rows[j] = k
        cols[j] = dofmap.cell_dofs(cell_id)
        vals[j] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    H = csr_matrix((vals, ij), shape=(nx, V.dim()))
    H.eliminate_zeros()
    if out == "scipy":
        return H
    elif out == "petsc":
        pH = Mat().createAIJ(size=H.shape, csr=(H.indptr, H.indices, H.data))
        return pH
    else:
        raise ValueError(f"out option {out} not recognised")


class SquareExpKronecker:
    def __init__(self, grid, scale, ell, nugget=1e-10):
        """Efficiently sample from a square exponential GP on 2D grid. """
        self.grid = grid
        self.n = self.grid.shape[0]
        self.x1 = np.unique(self.grid[:, 0])
        self.x2 = np.unique(self.grid[:, 1])

        self.K_x1 = sq_exp_covariance(self.x1[:, np.newaxis],
                                      scale=scale,
                                      ell=ell,
                                      nugget=nugget)
        self.K_x2 = sq_exp_covariance(self.x2[:, np.newaxis],
                                      scale=scale,
                                      ell=ell,
                                      nugget=nugget)

        self.K_chol_x1 = cholesky(self.K_x1, lower=True)
        self.K_chol_x2 = cholesky(self.K_x2, lower=True)

        self.P = None

    def set_permutation(self, V):
        """Set the permutation matrix to map from grid to DOF ordering. """
        PT = build_observation_operator(self.grid,
                                        V)  # PT: DOF -> grid ordering
        self.P = PT.T  # P: grid -> DOF ordering
        PPT = self.P @ self.P.T
        np.testing.assert_allclose(PPT.diagonal(), np.ones(self.P.shape[0]))

    def sample(self, n_sample=1):
        """Sample from GP using Kronecker structure (permuting if needed). """
        samples = np.zeros((self.n, n_sample))
        for i in range(n_sample):
            z = np.random.normal(size=(self.n, ))
            samples[:, i] = kron_matvec(self.K_chol_x1, self.K_chol_x2, z)

        if n_sample == 1:
            samples = samples.flatten()

        if self.P is None:
            return samples
        else:
            return self.P @ samples


if __name__ == "__main__":
    from scripts.poisson_2d import PoissonUnitTheta
    n = 100
    pois = PoissonUnitTheta(n)
    scale, ell = 1., 0.1

    gp = SquareExpKronecker(pois.mesh.coordinates(), scale, ell)

    sample_kron = gp.sample()
    plt.tricontourf(gp.grid[:, 0], gp.grid[:, 1], sample_kron, 64)
    plt.show()

    gp.set_permutation(pois.V)

    sample_fenics = fe.Function(pois.V)
    sample_fenics.vector()[:] = gp.P @ sample_kron
    im = fe.plot(sample_fenics)
    plt.colorbar(im)
    plt.show()
