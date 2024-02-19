import logging
import pyamg

import numpy as np
import fenics as fe
import jax.numpy as jnp

from functools import partial
from jax import grad, jit, jacfwd

from scipy.linalg import cholesky, cho_solve
from scipy.sparse import diags
from scipy.sparse.linalg import (cg, eigsh, LinearOperator, splu, spilu,
                                 SuperLU)
from sksparse.cholmod import analyze
from sksparse.cholmod import cholesky as spolesky

from .utils import (build_observation_operator, dolfin_to_csr,
                    sq_exp_covariance, SquareExpKronecker, sparse_to_jax,
                    A_jax_matvec)

from jax.config import config
config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
fe.set_log_level(40)


class LowRankFEMCov:
    def __init__(self, C_inv, nugget=1e-8, k=50):
        """
        Low-rank representation of the statFEM covariance matrix.

        Computed through the Cholesky of the sparse precision. Adds on a
        rank-correction term, so that the preconditioner is not rank-deficient.

        Parameters
        ----------
        C_inv: scipy.sparse matrix
            Sparse precision matrix.
        nugget: float, optional
            Diagonal correction term to ensure the matrix is full-rank.
        k: int, optional
            Number of eigenvalue/vector pairs to compute.
        """
        factor = analyze(C_inv, ordering_method="natural")
        C_inv_chol = factor.cholesky(C_inv)

        def matvec(x):
            return C_inv_chol.solve_A(x)

        C = LinearOperator(C_inv.shape, matvec)
        self.vals, self.vecs = eigsh(C, k, which="LM")
        self.vals_sqrt = np.sqrt(self.vals)

        self.nugget = nugget
        self.vals_diag = diags(self.vals)
        self.identity = diags(np.ones_like(self.vals))
        self.vals_sqrt_diag = diags(self.vals_sqrt)

        self.X = np.sqrt(self.nugget) * (
            np.sqrt(self.identity + self.vals_diag / self.nugget) -
            self.identity)
        logger.info(f"PC spectrum: {self.vals[0]:.6e} to {self.vals[-1]:.6e}")

    def matvec(self, x):
        """
        Matrix-vector product with x.

        Parameters
        ----------
        x: np.ndarray
        """
        temp = self.vecs.T @ x
        return self.nugget * x + self.vecs @ (self.vals_diag @ temp)

    def matvec_sqrt(self, x):
        """
        Matrix square-root-vector product with x.

        Parameters
        ----------
        x: np.ndarray
        """
        temp = self.vecs.T @ x
        return (np.sqrt(self.nugget) * x + self.vecs @ (self.X @ temp))


class LUPreconditioner:
    def __init__(self, A, G, G_sqrt, pc_type):
        """
        Preconditioning matrix using LU factors.

        Preconditioning options ('pc_type') are full LU ('lu'), scipy-default
        incomplete LU ('ilu'), and no fill-in iLU ('ilu_cheap').
        """
        self.G, self.G_sqrt = G, G_sqrt

        if pc_type == "ilu":
            self.A_factor = spilu(A.tocsc())
        elif pc_type == "ilu_cheap":
            self.A_factor = spilu(A.tocsc(), drop_tol=1e-8, fill_factor=1)
        else:
            print("Invalid LU preconditioning supplied")
            raise ValueError

    def matvec(self, x):
        AT_inv_x = self.A_factor.solve(x, trans="T")
        G_AT_inv_x = self.G @ AT_inv_x
        return self.A_factor.solve(G_AT_inv_x)

    def matvec_sqrt(self, x):
        temp = self.G_sqrt @ x
        return self.A_factor.solve(temp)


class PoissonUnit:
    def __init__(self, nx):
        self.mesh = fe.UnitSquareMesh(nx, nx)

        self.V = fe.FunctionSpace(self.mesh, "CG", 1)
        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        bc = fe.DirichletBC(self.V, fe.Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(bc.get_boundary_values().keys()))

        alpha = 1.
        a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        L = fe.inner(fe.Constant(alpha), v) * fe.dx

        A, b = fe.assemble_system(a, L, bc)
        self.A = dolfin_to_csr(A)
        self.b = b[:]
        self.u = np.zeros_like(self.b)
        self.A_lu = splu(self.A.tocsc())

    def setup_G(self, sigma):
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        m = fe.inner(u, v) * fe.dx
        M = fe.as_backend_type(fe.assemble(m))
        G_diag = sigma**2 * M.mat().getRowSum().array
        G_diag_inv = 1 / G_diag
        G_diag_sqrt = np.sqrt(G_diag)

        self.G = diags(G_diag)
        self.G_inv = diags(G_diag_inv)
        self.G_sqrt = diags(G_diag_sqrt)

        self.C_inv = self.A @ self.G_inv @ self.A

    def setup_pc(self, pc_type="ilu"):
        if "lu" in pc_type:
            self.M = LUPreconditioner(self.A, self.G, self.G_sqrt, pc_type)
        elif pc_type == "low_rank":
            self.M = LowRankFEMCov(self.A, self.G, self.G_inv, 50)
        elif pc_type == "amg":
            self.M = pyamg.smoothed_aggregation_solver(self.A, max_coarse=4)
        else:
            print("Not sure which PC to use")
            raise ValueError

        self.pc_type = pc_type

    def grad_phi(self, u):
        diff = self.A @ u - self.b
        return self.A.T @ (self.G_inv @ diff)

    def grad_phi_pc(self, u):
        """ M @ (grad_phi) """
        grad = self.grad_phi(u)
        return self.M.matvec(grad)

    def ula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        drift = self.grad_phi(self.u)

        u_next = self.u - eta * drift + np.sqrt(2 * eta) * z
        self.u = np.copy(u_next)

    def pula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        xi = self.M.matvec_sqrt(z)
        w = np.sqrt(2 * eta) * xi

        drift = self.grad_phi_pc(self.u)
        u_next = self.u - eta * drift + w
        self.u = np.copy(u_next)

    def pula_step_amg(self, eta=1e-2):
        assert self.pc_type == "amg"
        grad_phi = self.grad_phi(self.u)
        A_inv_grad_phi = self.M.solve(grad_phi, tol=1e-8, accel="cg")
        drift = eta * self.G @ A_inv_grad_phi

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        xi = self.G_sqrt @ z
        w = np.sqrt(2 * eta) * xi

        u_next = self.u - self.M.solve(drift + w, tol=1e-8, accel="cg")
        self.u[:] = u_next

    def exact_step(self):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        xi = self.G_sqrt @ z

        return self.A_lu.solve(self.b + xi)

    def generate_xi(self):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        fe.xi = self.G_sqrt @ z

    def u_mean(self):
        return self.A_lu.solve(self.b)


class PoissonBase:
    def __init__(self, nx):
        self.mesh = fe.UnitSquareMesh(nx, nx)
        self.V = fe.FunctionSpace(self.mesh, "CG", 1)

        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        # really the median theta, but keep for backwards compatibility
        x0, x1 = self.x_dofs[:, 0], self.x_dofs[:, 1]
        self.theta_mean = 1 + 0.3 * np.sin(np.pi * (x0 + x1))

        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        self.bc = fe.DirichletBC(self.V, fe.Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(self.bc.get_boundary_values().keys()))

        self.theta = fe.Function(self.V)
        self.theta.vector()[:] = self.theta_mean
        self.a = self.theta * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx

        alpha = 1.
        self.L = fe.inner(fe.Constant(alpha), v) * fe.dx

        b = fe.as_backend_type(fe.assemble(self.L))
        self.bc.apply(b)
        self.b = b.vec().array

        self.u = np.zeros((self.n_dofs, ))

    def assemble_A(self):
        """
        Assemble the a-form into self.A (symmetry-preserving).
        """
        A, b = fe.assemble_system(self.a, self.L, self.bc)
        self.A = dolfin_to_csr(fe.as_backend_type(A).mat())
        self.b = b[:]

    def residual_form(self, u):
        r = self.A @ u / 2 - self.b
        return np.dot(u, r)

    def grad_residual_form(self, u):
        return self.A @ u - self.b

    def gradient_step(self, u, stepsize):
        """
        Single step of gradient descent for the residual form: sanity check
        that things are reasonable.
        """
        grad = self.grad_residual_form(u)
        return u - stepsize * grad


class PoissonUnitTheta(PoissonBase):
    def __init__(self, nx):
        """
        StatFEM sampler for the prior, with stochastic diffusivity.

        Parameters
        ----------
        nx: int
            Number of FEM cells to compute over.
        """
        super().__init__(nx=nx)

    def setup_G(self, scale, dense=False, ell=1.):
        """
        Set up the statFEM G matrix. Computes the matrix, it's square-root, and
        implicitly computes the inverse.

        Parameters
        ----------
        scale: float
            Variance of the PDE RHS.
        dense: bool, optional
            Flag to use dense `G`. Defaults to false.
        ell: float, optional
            Optional length-scale (only used if `G` is dense).
        """
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        m = fe.inner(u, v) * fe.dx
        M = fe.as_backend_type(fe.assemble(m))
        M_scipy = dolfin_to_csr(M)

        G_diag = scale**2 * M.mat().getRowSum().array
        G_diag_inv = 1 / G_diag
        G_diag_sqrt = np.sqrt(G_diag)

        self.G = diags(G_diag)
        self.G_inv = diags(G_diag_inv)
        self.G_sqrt = diags(G_diag_sqrt)
        self.G_dense = None

        if dense:
            lower = True
            self.G_diag_approx = diags(G_diag)
            self.G_sqrt_sparse = self.G_sqrt.copy()
            self.G_inv_sparse = self.G_inv.copy()

            K = sq_exp_covariance(self.x_dofs, scale, ell)

            self.G_dense = M_scipy @ K @ M_scipy.T
            self.G_sqrt = cholesky(self.G_dense, lower=lower)

            # should give O(d^2) scaling
            def solve_dense_G(x):
                return cho_solve((self.G_sqrt, lower), x)

            self.G_inv = solve_dense_G(np.eye(self.n_dofs))

    def setup_theta(self, scale, ell, method="default", nugget=1e-10):
        """
        Setup the log-GP diffusivity theta.

        Parameters
        ----------
        scale: float
            GP variance parameter.
        ell: float
            GP length-scale parameter.
        method: str, optional
            Method by which we will use the GP. Default uses the dense
            covariance matrix. Kronecker stores the covariance as a Kronecker
            structure, which enables dramatic speedups.
        nugget: float, optional
            Diagonal correction term to add on to the GP covariance.
        """
        logger.info("starting theta setup")
        self.theta_method = method

        if method == "default":
            self.K_theta = sq_exp_covariance(self.x_dofs,
                                             scale=scale,
                                             ell=ell,
                                             nugget=nugget)
            self.K_theta_chol = cholesky(self.K_theta, lower=True)
        elif method == "kronecker":
            self.theta_gp = SquareExpKronecker(self.mesh.coordinates(),
                                               scale=scale,
                                               ell=ell,
                                               nugget=nugget)
            self.theta_gp.set_permutation(self.V)
        else:
            return ValueError(
                "theta_method incorrect: either 'default' or 'kronecker'")

        logger.info("theta setup all done")

    def sample_theta(self):
        """
        Sample theta into the Fenics function self.theta.
        """
        w = np.zeros_like(self.theta_mean)

        if self.theta_method == "default":
            z = np.random.normal(size=(self.n_dofs, ))
            w[:] = self.K_theta_chol @ z
        else:
            w[:] = self.theta_gp.sample()

        # theta is log-normally distributed
        self.theta.vector()[:] = np.copy(np.exp(np.log(self.theta_mean) + w))

    def setup_pc(self, pc_type, **kwargs):
        """
        Setup the preconditioner using the mean of theta.

        This does NOT mean that this preconditioner is used for the sampler
        (you choose the sampler). This uses a sparse `G` approximation,
        irrespective of whether `G` is dense or not.

        Parameters
        ----------
        pc_type: str
            Which preconditioner to set up. Options are lr, amg, lu, chol, ilu,
            diag.
        """
        logger.info("setting up preconditioner")
        self.pc_type = pc_type

        self.theta.vector()[:] = np.copy(self.theta_mean)
        self.assemble_A()

        if self.G_dense is not None:
            C_inv = self.A.T @ self.G_inv_sparse @ self.A
        else:
            C_inv = self.A.T @ self.G_inv @ self.A

        if pc_type == "lr":
            self.M = LowRankFEMCov(C_inv, **kwargs)
        elif pc_type == "amg":
            self.M = pyamg.smoothed_aggregation_solver(self.A, max_coarse=4)
        elif pc_type == "lu" or pc_type == "chol":
            logger.info("mean-theta preconditioning, now with CHOLMOD")
            self.M = analyze(self.A.tocsc())
            self.M = self.M.cholesky(self.A.tocsc())
        elif pc_type == "ilu":
            self.M = spilu(self.A.tocsc(), drop_tol=1e-12, fill_factor=1)
        elif pc_type == "diag":
            # use the diagonal of the precision to precondition
            pass
        else:
            raise ValueError(f"preconditioner {pc_type} not supported")

        logger.info("preconditioner setup all done")

    def log_target(self, u):
        """
        Log_target = -phi (unnormalized).

        Parameters
        ----------
        u: np.ndarray
            vector to evaluate the log-target at.
        """
        diff = self.A @ u - self.b
        return -np.dot(diff, self.G_inv @ diff) / 2

    def log_mala_prop(self, u, u_curr, eta):
        """
        Log-proposal density (unnormalized) for MALA.

        Parameters
        ----------
        u: np.ndarray
            Vector to evaluate the log-target at.
        u_curr: np.ndarray
            Previous solution vector to evaluate the log-target at.
        eta: float
            Stepsize parameter.
        """
        grad_phi = self.grad_phi(u_curr)
        mean = u_curr - eta * grad_phi
        return -np.dot(u - mean, u - mean) / (4 * eta)

    def log_pmala_prop(self, u, u_curr, eta, A_factor):
        """
        Log-proposal density (unnormalized as determinants cancel).

        Parameters
        ----------
        u: np.ndarray
            Vector to evaluate the log-target at.
        u_curr: np.ndarray
            Previous solution vector to evaluate the log-target at.
        eta: float
            Stepsize parameter.
        A_factor: SuperLU factor or pyAMG factor.
            Factorization of stiffness matrix, with a `solve` method.
        """
        if type(A_factor) == SuperLU:
            A_inv_b = A_factor.solve(self.b)
        else:
            A_inv_b = A_factor.solve(self.b, tol=1e-8, accel="cg")

        mean = (1 - eta) * u_curr + eta * A_inv_b
        A_diff = self.A @ (u - mean)
        return -np.dot(A_diff, self.G_inv @ A_diff) / (4 * eta)

    def grad_phi(self, u):
        """
        Gradient of the log-target.

        Parameters
        ----------
        u: np.ndarray
            Vector to evaluate the log-target at.
        """
        diff = self.A @ u - self.b
        return self.A.T @ (self.G_inv @ diff)

    def ula_step(self, eta=1e-2, fixed_theta=False):
        """
        ULA step.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        grad = self.grad_phi(self.u)
        u_next = self.u - eta * grad + np.sqrt(2 * eta) * z

        self.u[:] = np.copy(u_next)

    def pula_step_exact(self, eta=1e-2, method="lu", fixed_theta=False):
        """
        ULA w/exact Hessian PC.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        rhs = eta * self.b + np.sqrt(2 * eta) * self.G_sqrt @ z

        if method == "lu":
            A_lu = splu(self.A.tocsc())
            w = A_lu.solve(rhs)
        elif method == "amg":
            M = pyamg.smoothed_aggregation_solver(self.A, max_coarse=4)
            w = M.solve(rhs, tol=1e-8, accel="cg")
        elif method == "krylov":
            A_ilu = spilu(self.A.tocsc(), drop_tol=1e-8, fill_factor=10)
            M = LinearOperator(self.A.shape, A_ilu.solve)

            w, info = cg(self.A, rhs, tol=1e-8, M=M)

        u_next = (1 - eta) * self.u + w
        self.u = np.copy(u_next)

    def pula_step_lr(self, eta=1e-2, fixed_theta=False):
        """
        ULA w/low-rank Hessian PC. """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        grad_phi = self.grad_phi(self.u)
        grad_phi_pc = self.M.matvec(grad_phi)
        w = np.sqrt(2 * eta) * self.M.matvec_sqrt(z)

        u_next = self.u - eta * grad_phi_pc + w
        self.u = np.copy(u_next)

    def pula_step_lu_mean(self, eta=1e-2, fixed_theta=False):
        """
        ULA w/mean-approximated A.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        w = np.sqrt(2 * eta) * self.G_sqrt @ z

        grad = self.grad_phi(self.u)
        A_inv_grad = self.M.solve_A(grad)
        drift = self.G @ A_inv_grad
        adjust = self.M.solve_A(-eta * drift + w)
        u_next = self.u + adjust
        self.u[:] = np.copy(u_next)

    def pula_step_ilu_mean(self, eta=1e-2, fixed_theta=False):
        """
        ULA w/mean-approximated incomplete factor of A.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        w = np.sqrt(2 * eta) * self.G_sqrt @ z

        grad = self.grad_phi(self.u)
        A_inv_grad = self.M.solve(grad)
        drift = self.G @ A_inv_grad

        adjust = self.M.solve(-eta * drift + w)
        u_next = self.u + adjust
        self.u[:] = np.copy(u_next)

    def pula_step_amg(self, eta=1e-2, fixed_theta=False):
        """
        ULA step w/AMG-computed Hessian PC.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        grad_phi = self.grad_phi(self.u)
        A_inv_grad_phi = self.M.solve(grad_phi, tol=1e-8, accel="cg")
        drift = eta * self.G @ A_inv_grad_phi

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        xi = self.G_sqrt @ z
        w = np.sqrt(2 * eta) * xi

        self.u -= self.M.solve(drift + w, tol=1e-8, accel="cg")

    def mala_step(self, eta=1e-2, fixed_theta=False):
        """
        Metropolis-adjusted Langevin.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        grad_phi = self.grad_phi(self.u)
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_prop = self.u - eta * grad_phi + np.sqrt(2 * eta) * z

        log_target_prop = self.log_target(u_prop)
        log_target_curr = self.log_target(self.u)

        log_prop = self.log_mala_prop(u_prop, self.u, eta)
        log_prop_flip = self.log_mala_prop(self.u, u_prop, eta)

        log_alpha = (log_target_prop - log_target_curr + log_prop_flip -
                     log_prop)

        if np.log(np.random.uniform()) <= log_alpha:
            self.u = np.copy(u_prop)
            accepted = True
        else:
            accepted = False

        return accepted

    def pmala_step(self, eta=1e-2, fixed_theta=False):
        """
        Metropolis-adjusted Langevin w/exact Hessian PC.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()
            self.A_factor = splu(self.A.tocsc())

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        w = self.A_factor.solve(eta * self.b +
                                np.sqrt(2 * eta) * self.G_sqrt @ z)

        u_prop = (1 - eta) * self.u + w
        log_target_prop = self.log_target(u_prop)
        log_target_curr = self.log_target(self.u)

        log_prop = self.log_pmala_prop(u_prop, self.u, eta, self.A_factor)
        log_prop_flip = self.log_pmala_prop(self.u, u_prop, eta, self.A_factor)

        log_alpha = (log_target_prop - log_target_curr + log_prop_flip -
                     log_prop)

        if np.log(np.random.uniform()) <= log_alpha:
            self.u = np.copy(u_prop)
            accepted = True
        else:
            accepted = False

        return accepted

    def exact_step(self, method="lu"):
        """
        Exact sample.

        Parameters
        ----------
        method: str, optional
            Method to sample with. Either `lu` or `amg`.
        """
        self.sample_theta()
        self.assemble_A()

        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.
        xi = self.G_sqrt @ z
        rhs = self.b + xi

        if method == "lu":
            self.A_lu = splu(self.A.tocsc())
            return self.A_lu.solve(rhs)
        elif method == "amg":
            M = pyamg.smoothed_aggregation_solver(self.A, max_coarse=4)
            return M.solve(rhs, tol=1e-8, accel="cg")
        else:
            raise ValueError(f"method {method} not recognised")


class PoissonUnitThetaPosterior(PoissonUnitTheta):
    def __init__(self, nx):
        """
        StatFEM sampler for the posterior, with stochastic diffusivity.

        Parameters
        ----------
        nx: int
            Number of FEM cells to compute over.
        """
        super().__init__(nx=nx)

    def setup_dgp(self,
                  x_obs,
                  n_obs,
                  sigma,
                  scale_factor=1.,
                  obs_function=None):
        """
        Set the simulation settings and preconditioner.

        Parameter
        ---------
        x_obs: np.ndarray
            Array of observation locations.
        n_obs: int
            Number of observation vectors.
        sigma: float
            Observational noise parameter.
        scale_factor: float, optional
            Factor to scale generated vectors by (used to induce model
            mismatch).
        """
        assert x_obs.shape[1] == 2

        self.x_obs = x_obs
        self.n_y = x_obs.shape[0]
        self.n_obs = n_obs

        self.H = build_observation_operator(x_obs, self.V)

        self.sigma_y = sigma
        self.R = diags(sigma**2 * np.ones(x_obs.shape[0]))
        self.R_sqrt = diags(sigma * np.ones(x_obs.shape[0]))
        self.R_inv = diags(1 / sigma**2 * np.ones(x_obs.shape[0]))

        self.theta.vector()[:] = np.copy(self.theta_mean)
        self.assemble_A()

        # setup preconditioning
        self.M = (self.A.T @ self.G_inv @ self.A +
                  self.n_obs * self.H.T @ self.R_inv @ self.H)
        self.factor = analyze(self.M, ordering_method="natural")
        self.M_chol = self.factor.cholesky(self.M)

        self.data_scale_factor = scale_factor

        if obs_function is None:
            self.obs_function = lambda x: x
        else:
            self.obs_function = obs_function

        self.y = np.zeros((self.n_y, self.n_obs))
        logger.info("Observing %d observations at %d locations", self.n_obs,
                    self.n_y)

    def setup_pc_post(self, u):
        """
        Setup preconditioner for nonlinear observations.

        Uses the Cholesky of the sparse precision matrix.
        """
        self.theta.vector()[:] = np.copy(self.theta_mean)
        self.assemble_A()

        # HACK: currently uses dense jacobian, then converts to sparse
        grad_obs_function = jacfwd(self.obs_function)
        J_obs_function_dense = grad_obs_function(self.H @ u)
        J_obs_function = diags(np.array(J_obs_function_dense.diagonal()))
        scaled_J = J_obs_function.T @ self.R_inv @ J_obs_function

        self.M = (self.A.T @ self.G_inv @ self.A +
                  self.n_obs * self.H.T @ scaled_J @ self.H)
        self.M_chol = spolesky(self.M.tocsc(), ordering_method="natural")

    def generate_data(self):
        """
        Generate a set of data, given the current set of data-generating
        parameters.
        """
        u_sample = np.zeros((self.n_dofs, self.n_obs))

        for i in range(self.n_obs):
            self.sample_theta()
            self.assemble_A()

            z = np.random.normal(size=(self.n_dofs, ))
            z[self.bc_dofs] = 0.
            xi = self.G_sqrt @ z
            rhs = self.b + xi

            A_lu = splu(self.A.tocsc())
            u_sample[:, i] = A_lu.solve(rhs)

        eta = self.sigma_y * np.random.normal(size=(self.n_y, self.n_obs))

        if callable(self.obs_function):
            y_true = self.obs_function(self.H @ u_sample)
        else:
            logger.error("Not sure how to generate observations")
            raise ValueError

        self.y[:] = self.data_scale_factor * y_true + eta
        self.y_mean = np.sum(self.y, axis=1) / self.n_obs

    def setup_jax(self):
        """
        Setup JAX for this problem. This implicitly ensures that all samplers
        will use JAX, under the hood, for gradients of the log-likelihood.
        """
        logger.info("Now using JAX for log-likelihoods and their gradients")

        self.use_jax = True
        self.H_jax = sparse_to_jax(self.H)
        self.u_obs = jnp.zeros((self.n_y, ))
        self.grad_log_likelihood_jax = jit(grad(self.log_likelihood_jax, 2))

    def load_data(self, y):
        """
        Load a pre-generated dataset.

        Parameters
        ----------
        y: np.ndarray
            Array of observations.
        """
        self.y[:] = y
        assert self.y.shape == (self.n_y, self.n_obs)

        self.y_mean = np.sum(self.y, axis=1) / self.n_obs
        assert self.y_mean.shape == (self.n_y, )

    def compute_mean_theta_prior(self):
        """
        Compute the prior where `theta` is fixed to its mean.
        """
        self.theta.vector()[:] = self.theta_mean
        self.assemble_A()

        self.A_lu = splu(self.A.tocsc())

        if self.G_dense is None:
            G_dense = np.squeeze(np.asarray(self.G.todense()))
        else:
            G_dense = self.G_dense

        mean = self.A_lu.solve(self.b)
        temp = self.A_lu.solve(G_dense)
        C = self.A_lu.solve(temp.T)

        return mean, C

    def compute_mean_theta_posterior(self):
        """
        Compute the posterior where `theta` is fixed to its mean.
        """
        logger.warning("Returns dense covariance matrix: be careful!")
        self.theta.vector()[:] = self.theta_mean
        self.assemble_A()

        C_inv = self.compute_precision()
        C = self.factor.solve_A(np.eye(self.n_dofs))

        mean_rhs = (self.n_obs * self.H.T @ self.R_inv @ self.y_mean +
                    self.A.T @ (self.G_inv @ self.b))
        mean = self.factor.solve_A(mean_rhs)

        return mean, C

    # TODO: rename this function to state linear requirement
    def sample_posterior_exact(self):
        """
        Sample the posterior; efficient due to sparsity of the precision.
        """
        logger.warning(
            "This samples from the exact posterior for the linear case:" +
            " y = Hu + e! Be careful!")
        self.sample_theta()
        self.assemble_A()

        if self.use_jax:
            pass

        C_inv = self.compute_precision()
        mean_rhs = (self.n_obs * self.H.T @ self.R_inv @ self.y_mean +
                    self.A.T @ (self.G_inv @ self.b))
        mean = self.factor.solve_A(mean_rhs)

        z = np.random.normal(size=(self.n_dofs, ))
        w = self.factor.solve_Lt(z, use_LDLt_decomposition=False)
        return mean + w

    def gradient_step(self, u, stepsize):
        """
        Single step of gradient descent for the (log) posterior.
        """
        grad_phi = self.grad_phi(u)
        return u - stepsize * grad_phi

    def adam_step(self,
                  u,
                  m,
                  v,
                  i,
                  stepsize,
                  beta1=0.9,
                  beta2=0.999,
                  eps=1e-8):
        """
        Single step of Adam for the ULA (log) posterior.

        Parameters
        ----------
        """
        grad_phi = self.grad_phi(u)
        m_next = beta1 * m + (1 - beta1) * grad_phi
        v_next = beta2 * v + (1 - beta2) * grad_phi**2

        m_hat = m_next / (1 - beta1**i)
        v_hat = v_next / (1 - beta2**i)

        u_next = u - stepsize * m_hat / (np.sqrt(v_hat) + eps)

        return u_next, m_next, v_next

    def newton_step(self, u, stepsize):
        """
        Single step of Gauss-Newton for the log-posterior.

        Efficient because the GN Hessian is sparse, in this case.
        """
        # TODO: refactor into a single function to avoid copy and paste
        grad_obs_function = jacfwd(self.obs_function)
        J_obs_function_dense = grad_obs_function(self.H @ u)
        J_obs_function = diags(np.array(J_obs_function_dense.diagonal()))
        scaled_J = J_obs_function.T @ self.R_inv @ J_obs_function

        H = (self.A.T @ self.G_inv @ self.A +
             self.n_obs * self.H.T @ scaled_J @ self.H)
        H_chol = spolesky(H.tocsc(), ordering_method="natural")

        grad = self.grad_phi(u)
        delta = H_chol.solve_A(grad)
        u_next = u - stepsize * delta

        return u_next, delta

    def euler_step(self, eta, pc=False):
        """
        Single step of the Euler-Maruyama for the Langevin diffusion.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        grad_phi = self.grad_phi(self.u)
        z = np.random.normal(size=(self.n_dofs, ))

        if pc:
            grad_phi_pc = self.M_chol.solve_A(grad_phi)
            w = self.M_chol.solve_Lt(z, use_LDLt_decomposition=False)
            u_next = self.u - eta * grad_phi_pc + np.sqrt(2 * eta) * w
        else:
            u_next = self.u - eta * grad_phi + np.sqrt(2 * eta) * z

        return u_next

    def ula_step(self, eta=1e-2, fixed_theta=False, pc=False):
        """
        Single step of the unadjusted Langevin algorithm.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        self.u[:] = self.euler_step(eta, pc=pc)

    def pula_step(self, eta=1e-2, fixed_theta=False):
        """
        Alias function for the preconditioned-ULA samplers. Mimics the prior.
        """
        self.ula_step(eta, fixed_theta, pc=True)

    def mala_step(self, eta=1e-2, fixed_theta=False, pc=False):
        """
        Single step of the Metropolis-adjusted langevin (preconditioning).

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool, optional
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        pc: bool, optional
            Whether or not to use preconditioning. If so, the mean-theta
            Hessian of the linear problem is used.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()

        u_prop = self.euler_step(eta, pc=pc)

        log_target_curr = self.log_target(self.u)
        log_target_prop = self.log_target(u_prop)

        log_q = self.log_mala_prop(u_prop, self.u, eta, pc)
        log_q_flip = self.log_mala_prop(self.u, u_prop, eta, pc)

        log_alpha = log_target_prop - log_target_curr + log_q_flip - log_q

        if np.log(np.random.uniform()) <= np.amin([0, log_alpha]):
            self.u[:] = u_prop
            return True
        else:
            return False

    def pmala_step(self, eta=1e-2, fixed_theta=False):
        """
        Alias function for the preconditioned-MALA samplers; mimics the prior.
        """
        self.mala_step(eta, fixed_theta, pc=True)

    def pcn_step(self, eta=1e-2, fixed_theta=False):
        """
        Preconditioned Crank-Nicolson algorithm. Uses the Cholesky
        factorization to sample from the prior.

        Parameters
        ----------
        eta: float
            Stepsize parameter.
        fixed_theta: bool
            Whether `theta` is sampled, or, reused from the previous
            iterations; if reused, the stiffness matrix is not assembled.
        """
        if not fixed_theta:
            self.sample_theta()
            self.assemble_A()
            self.A_factor = analyze(self.A.tocsc())
            self.A_factor = self.A_factor.cholesky(self.A.tocsc())

        mean_prior = self.A_factor.solve_A(self.b)

        z = np.random.normal(size=(self.n_dofs, ))
        w = eta * self.A_factor.solve_A(self.G_sqrt @ z)

        u_prop = (mean_prior + np.sqrt(1 - eta**2) * (self.u - mean_prior) + w)

        log_likelihood_prop = self.log_likelihood_jax(self.y, self.H_jax,
                                                      u_prop, self.u_obs)
        self.u_obs.at[:].set(0.0)
        log_likelihood_curr = self.log_likelihood_jax(self.y, self.H_jax,
                                                      self.u, self.u_obs)
        self.u_obs.at[:].set(0.0)

        log_alpha = log_likelihood_prop - log_likelihood_curr

        if np.log(np.random.uniform()) <= np.min([0, log_alpha]):
            self.u[:] = u_prop
            return True
        else:
            return False

    def log_likelihood(self, u):
        """
        Log likelihood (up to prop. constant).

        Parameters
        ----------
        u: np.ndarray
            vector to evaluate the log-likelihood at.
        """
        resid = (self.y.T - self.H @ u).T  # transpose to make array commute
        return -np.sum(resid**2) / (2 * self.sigma_y**2)

    @partial(jit, static_argnums=(0, ))
    def log_likelihood_jax(self, y, H, u, u_obs):
        """ Log likelihood, with arb. observations, in jax. """
        u_obs = A_jax_matvec(H, u, u_obs)  # Hu = u_obs
        return jnp.sum(-(y.T - self.obs_function(u_obs))**2 /
                       (2 * self.sigma_y**2))

    def log_target(self, u):
        """
        Log posterior (up to prop. constant).

        Parameters
        ----------
        u: np.ndarray
            vector to evaluate the log-target at.
        """
        diff = self.A @ u - self.b
        log_prior = -np.dot(diff, self.G_inv @ diff) / 2

        if self.use_jax:
            ll = self.log_likelihood_jax(self.y, self.H_jax, u, self.u_obs)
            self.u_obs.at[:].set(0.0)  # need to reset afterwards
        else:
            ll = self.log_likelihood(self.u)

        return log_prior + ll

    def grad_phi(self, u):
        """
        Gradient of the negative log posterior.

        Parameters
        ----------
        u: np.ndarray
            vector to evaluate the log-posterior at.
        """
        diff = self.A @ u - self.b

        if self.use_jax:
            ll_grad = self.grad_log_likelihood_jax(self.y, self.H_jax, u,
                                                   self.u_obs)
            self.u_obs.at[:].set(0.0)  # need to reset afterwards
        else:
            resid = self.y_mean - self.H @ u
            ll_grad = self.n_obs * self.H.T @ (self.R_inv @ resid)

        return self.A.T @ (self.G_inv @ diff) - ll_grad

    def log_mala_prop(self, u, u_curr, eta, pc=False):
        """
        Log-proposal density (unnormalized) for MALA. Computes q(u | u_curr).

        Parameters
        ----------
        u: np.ndarray
            Vector to evaluate the log-target at.
        u_curr: np.ndarray
            Previous solution vector to evaluate the log-target at.
        eta: float
            Stepsize parameter.
        pc: bool, optional
            Whether or not preconditioning has been applied.
        """
        grad_phi = self.grad_phi(u_curr)

        if pc:
            mean = u_curr - eta * self.M_chol.solve_A(grad_phi)
            return -np.dot(u - mean, self.M @ (u - mean)) / (4 * eta)
        else:
            mean = u_curr - eta * grad_phi
            return -np.dot(u - mean, u - mean) / (4 * eta)

    def compute_precision(self):
        """
        Compute the precision and update the cholesky.
        """
        C_inv = (self.A.T @ self.G_inv @ self.A +
                 self.n_obs * self.H.T @ self.R_inv @ self.H)
        self.factor.cholesky_inplace(C_inv)
        return C_inv

    def sample_prior_exact(self):
        """
        Helper function to sample from the prior (uses the LU decomposition).
        """
        self.sample_theta()
        self.assemble_A()

        self.A_lu = splu(self.A.tocsc())

        z = np.random.normal(size=(self.n_dofs, ))
        return self.A_lu.solve(self.b + self.G_sqrt @ z)


class NonlinearPoisson1D:
    def __init__(self, nx):
        self.mesh = fe.UnitIntervalMesh(nx)
        self.V = fe.FunctionSpace(self.mesh, "CG", 1)
        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        v = fe.TestFunction(self.V)
        self.u = fe.Function(self.V)
        self.bc = fe.DirichletBC(self.V, fe.Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(self.bc.get_boundary_values().keys()))

        self.xi = fe.Function(self.V)
        self.f = fe.interpolate(
            fe.Expression("8 * sin(3 * pi * pow(x[0], 2))", degree=4), self.V)

        self.F = ((1 + self.u**2) * fe.inner(fe.grad(self.u), fe.grad(v)) -
                  fe.inner(self.f, v) - fe.inner(self.xi, v)) * fe.dx
        self.J = fe.derivative(self.F, self.u, fe.TrialFunction(self.V))

        L = fe.inner(self.f, v) * fe.dx
        self.b = fe.assemble(L)
        self.bc.apply(self.b)

        problem = fe.NonlinearVariationalProblem(self.F,
                                                 self.u,
                                                 bcs=self.bc,
                                                 J=self.J)
        self.solver = fe.NonlinearVariationalSolver(problem)
        prm = self.solver.parameters
        prm["newton_solver"]["linear_solver"] = "gmres"
        prm["newton_solver"]["preconditioner"] = "petsc_amg"

        self.u_curr = np.zeros((self.n_dofs, ))
        self.J_PC_lu = None

    def linear_solve_xi_test(self, rtol=1e-10):
        """Test that Scipy and Fenics solve align to relative tolerance rtol.

        Verifies that premultiplying by M^{-1} does in fact scale as needed.
        """
        z = self.generate_z()

        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        L = fe.inner(self.f, v) * fe.dx

        A, b = fe.assemble(a), fe.assemble(L)
        self.bc.apply(A, b)
        A, b = dolfin_to_csr(A), b[:]
        A_lu = splu(A.tocsc())

        xi = self.G_sqrt @ z
        xi[self.bc_dofs] = 0.
        u_scipy = A_lu.solve(b + xi)

        xi = fe.Function(self.V)
        xi.vector()[:] = np.copy(self.M_lu.solve(self.G_sqrt @ z))
        L += fe.inner(xi, v) * fe.dx

        u = fe.Function(self.V)
        fe.solve(a == L, u, bcs=self.bc)
        u_fenics = u.vector()[:]

        norm = np.linalg.norm
        rel_diff = norm(u_scipy - u_fenics) / norm(u_fenics)
        assert rel_diff < rtol
        logger.info("rel diff: %e", rel_diff)

    def setup_G(self, sigma):
        """Set up the statFEM G matrix. """
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        m = fe.inner(u, v) * fe.dx
        M = fe.as_backend_type(fe.assemble(m))
        M_scipy = dolfin_to_csr(M)
        self.M_lu = splu(M_scipy.tocsc())

        G_diag = sigma**2 * M.mat().getRowSum().array
        G_diag_inv = 1 / G_diag
        G_diag_sqrt = np.sqrt(G_diag)

        self.G = diags(G_diag)
        self.G_inv = diags(G_diag_inv)
        self.G_sqrt = diags(G_diag_sqrt)

    def setup_pc(self):
        self.solver.solve()
        u_solve = self.u.vector()[:]
        F, J = self.assemble_system(u_solve)

        self.M_inv = J.T @ self.G_inv @ J
        self.M_inv_chol = analyze(self.M_inv, ordering_method="natural")
        self.M_inv_chol.cholesky_inplace(self.M_inv)

    def assemble_system(self, u):
        self.u.vector()[:] = u
        J, F = fe.assemble_system(self.J, self.F, bcs=self.bc)
        return F[:], dolfin_to_csr(J)

    def compute_hessian_map(self):
        """ Evaluate the Hessian matrix at the MAP. """
        self.solver.solve()
        F, J = self.assemble_system(self.u.vector()[:])
        J_lu = splu(J.tocsc())

        H = (J.T @ self.G_inv @ J).todense()
        for i in range(self.n_dofs):
            d = fe.Function(self.V)
            d.vector()[i] = 1.

            # derivative in two directions
            ddJ = fe.assemble(
                fe.derivative(fe.derivative(self.J, self.u, d), self.u,
                              d)).array()
            H[i, i] -= np.trace(J_lu.solve(ddJ))

            for j in range(self.n_dofs):
                e = fe.Function(self.V)
                e.vector()[j] = 1.

                # derivative in each direction
                dJ_first_direction = fe.assemble(
                    fe.derivative(self.J, self.u, d)).array()
                dJ_second_direction = fe.assemble(
                    fe.derivative(self.J, self.u, e)).array()

                J_inv_first_direction = J_lu.solve(dJ_first_direction)
                J_inv_second_direction = J_lu.solve(dJ_second_direction)

                trace_first_order = np.sum(J_inv_first_direction *
                                           J_inv_second_direction.T)

                H[i, j] -= trace_first_order

        return H

    def compute_trace_derivative(self, J_factor):
        # TODO: clean up this loop
        tr = np.zeros((self.n_dofs, ))
        for i in range(self.n_dofs):
            d = fe.Function(self.V)
            d.vector()[i] = 1.

            H = fe.assemble(fe.derivative(self.J, self.u, d))
            H_vec = H.array()[:, i]

            J_inv_H = J_factor.solve(H_vec)
            tr[i] = J_inv_H[i]

        return tr

    def grad_phi(self, u):
        F, J = self.assemble_system(u)
        J_lu = splu(J.tocsc())

        trace_term = self.compute_trace_derivative(J_lu)
        return -trace_term + J.T @ self.G_inv @ F

    def log_target(self, u):
        F, J = self.assemble_system(u)
        J_lu = splu(J.tocsc())
        log_det = np.sum(np.log(J_lu.L.diagonal()) + np.log(J_lu.U.diagonal()))
        return -log_det + F.T @ self.G_inv @ F / 2

    def ula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        u_curr -= eta * self.grad_phi(u_curr) + np.sqrt(2 * eta) * z
        self.u_curr[:] = u_curr

    def pula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        # apply preconditioner
        grad_phi = self.grad_phi(self.u_curr)
        grad_phi_pc = self.M_inv_chol.solve_A(grad_phi)
        w = self.M_inv_chol.solve_Lt(z, use_LDLt_decomposition=False)

        self.u_curr[:] -= (eta * grad_phi_pc + np.sqrt(2 * eta) * w)

    def tula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        grad_phi = self.grad_phi(u_curr)
        gradnorm = np.linalg.norm(grad_phi)
        u_curr -= eta * (1 / (1 + eta * gradnorm)) * grad_phi + np.sqrt(
            2 * eta) * z
        self.u_curr[:] = u_curr

    def tulac_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        grad_phi = self.grad_phi(u_curr)
        gradnorm = np.abs(grad_phi)
        u_curr -= eta * (1 / (1 + eta * gradnorm)) * grad_phi + np.sqrt(
            2 * eta) * z
        self.u_curr[:] = u_curr

    def exact_sample(self):
        """Generate exact sample with Fenics. """
        self.generate_xi(scale_Minv=True)
        self.solver.solve()
        return self.u.vector()[:]

    def generate_xi(self, scale_Minv=False):
        """Generate xi and set it in the nonlinear form. """
        z = self.generate_z()
        xi = self.G_sqrt @ z

        if scale_Minv:
            xi = self.M_lu.solve(xi)

        self.xi.vector()[:] = np.copy(xi)

    def generate_z(self):
        """Return a std. normal vector of length `self.n_dofs`. """
        return np.random.normal(size=(self.n_dofs, ))

    def zero_u_curr(self):
        """Helper function to set all elements of u_curr to 0. """
        assert self.u_curr.shape == (self.n_dofs, )
        self.u_curr[:] = 0.

    def first_order_approx(self):
        """Compute approximation to prior measure using Frechet derivative of
        weak form; linearized about the deterministic solution.
        """
        logger.warning("Returns DENSE COVARIANCE MATRIX --- be careful")

        self.xi.vector()[:] = 0.
        fe.solve(self.F == 0, self.u, bcs=self.bc, J=self.J)
        mean = np.copy(self.u.vector()[:])

        J = fe.assemble(self.J)
        self.bc.apply(J)
        J_scipy = dolfin_to_csr(J)
        J_scipy_lu = splu(J_scipy.tocsc())

        temp = J_scipy_lu.solve(self.G.todense())
        cov = J_scipy_lu.solve(temp.T)

        return mean, cov
