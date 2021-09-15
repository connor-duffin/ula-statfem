import logging

import numpy as np
import fenics as fe

import pyamg

from scipy.linalg import cholesky
from scipy.sparse import diags
from scipy.sparse.linalg import (cg, eigsh, LinearOperator, splu, spilu,
                                 SuperLU)
from sksparse.cholmod import analyze

from .utils import (build_observation_operator, dolfin_to_csr,
                    sq_exp_covariance, SquareExpKronecker)

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


class PoissonUnitTheta:
    def __init__(self, nx):
        """
        StatFEM sampler for the prior, with stochastic diffusivity.

        Parameters
        ----------
        nx: int
            Number of FEM cells to compute over.
        """
        self.mesh = fe.UnitSquareMesh(nx, nx)

        self.V = fe.FunctionSpace(self.mesh, "CG", 1)
        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        self.bc = fe.DirichletBC(self.V, fe.Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(self.bc.get_boundary_values().keys()))

        self.theta = fe.Function(self.V)
        self.a = self.theta * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx

        alpha = 1.
        self.L = fe.inner(fe.Constant(alpha), v) * fe.dx

        b = fe.as_backend_type(fe.assemble(self.L))
        self.bc.apply(b)

        self.b = b.vec().array
        self.u = np.zeros_like(self.b)

        # really the median theta, but keep for backwards compatibility
        x0, x1 = self.x_dofs[:, 0], self.x_dofs[:, 1]
        self.theta_mean = 1 + 0.3 * np.sin(np.pi * (x0 + x1))

    def setup_G(self, sigma):
        """
        Set up the statFEM G matrix.

        Parameters
        ----------
        sigma: float
            Variance of the PDE RHS.
        """
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
        (you choose the sampler).

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

    def assemble_A(self):
        """
        Assemble the a-form into self.A (symmetry-preserving).
        """
        A, b = fe.assemble_system(self.a, self.L, self.bc)
        self.A = dolfin_to_csr(fe.as_backend_type(A).mat())
        self.b = b[:]

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
        A_factor: SuperLU factor or AMG factor.
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

    def setup_dgp(self, x_obs, n_obs, sigma, scale_factor=1.):
        """
        Set the simulation settings for the data generating process.

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

        self.data_scale_factor = scale_factor

        self.y = np.zeros((self.n_y, self.n_obs))
        logger.info("Observing %d observations at %d locations", self.n_obs,
                    self.n_y)

    def setup_pc_post(self):
        """
        Setup preconditioning using mean-theta stiffness.

        Uses the Cholesky of the sparse precision matrix.
        """
        self.theta.vector()[:] = np.copy(self.theta_mean)
        self.assemble_A()

        C_inv = (self.A.T @ self.G_inv @ self.A +
                 self.n_obs * self.H.T @ self.R_inv @ self.H)
        self.factor = analyze(C_inv, ordering_method="natural")

        self.M = C_inv
        self.M_chol = self.factor.cholesky(C_inv)

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
        self.y[:] = self.data_scale_factor * self.H @ u_sample + eta
        self.y_mean = np.sum(self.y, axis=1) / self.n_obs

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

    def sample_posterior_exact(self):
        """
        Sample the posterior; efficient due to sparsity of the precision.
        """
        self.sample_theta()
        self.assemble_A()

        C_inv = self.compute_precision()
        mean_rhs = (self.n_obs * self.H.T @ self.R_inv @ self.y_mean +
                    self.A.T @ (self.G_inv @ self.b))
        mean = self.factor.solve_A(mean_rhs)

        z = np.random.normal(size=(self.n_dofs, ))
        w = self.factor.solve_Lt(z, use_LDLt_decomposition=False)
        return mean + w

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

    def ula_step(self, eta=1e-2, fixed_theta=False):
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

        self.u[:] = self.euler_step(eta, pc=False)

    def pula_step(self, eta=1e-2, fixed_theta=False):
        """
        Single step of the preconditioned unadjusted Langevin algorithm.

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

        self.u[:] = self.euler_step(eta, pc=True)

    def mala_step(self, eta=1e-2, fixed_theta=False):
        """
        Metropolis-adjusted langevin (no preconditioning).

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

        u_prop = self.euler_step(eta, pc=False)

        log_target_curr = self.log_target(self.u)
        log_target_prop = self.log_target(u_prop)

        log_q = self.log_mala_prop(u_prop, self.u, eta)
        log_q_flip = self.log_mala_prop(self.u, u_prop, eta)

        log_alpha = log_target_prop - log_target_curr + log_q_flip - log_q

        if np.log(np.random.uniform()) <= np.amin([0, log_alpha]):
            self.u[:] = u_prop
            return True
        else:
            return False

    def pmala_step(self, eta=1e-2, fixed_theta=False):
        """
        Preconditioned Metropolis-adjusted langevin.

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

        u_prop = self.euler_step(eta, pc=True)

        log_target_curr = self.log_target(self.u)
        log_target_prop = self.log_target(u_prop)

        log_q = self.log_pmala_prop(u_prop, self.u, eta)
        log_q_flip = self.log_pmala_prop(self.u, u_prop, eta)

        log_alpha = log_target_prop - log_target_curr + log_q_flip - log_q

        if np.log(np.random.uniform()) <= np.amin([0, log_alpha]):
            self.u[:] = u_prop
            return True
        else:
            return False

    def pcn_step(self, eta=1e-2, fixed_theta=False):
        """
        Preconditioned Crank-Nicolson algorithm. Uses the LU factorization to
        solve the system at each iteration.

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

        log_likelihood_prop = self.log_likelihood(u_prop)
        log_likelihood_curr = self.log_likelihood(self.u)

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
        log_likelihood = self.log_likelihood(u)
        return log_prior + log_likelihood

    def log_mala_prop(self, u, u_curr, eta):
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
        """
        grad_phi = self.grad_phi(u_curr)
        mean = u_curr - eta * grad_phi
        return -np.dot(u - mean, u - mean) / (4 * eta)

    def log_pmala_prop(self, u, u_curr, eta):
        """
        Log-proposal density (unnormalized) for preconditioned-MALA. Computes
        q(u | u_curr).

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
        mean = u_curr - eta * self.M_chol.solve_A(grad_phi)
        return -np.dot(u - mean, self.M @ (u - mean)) / (4 * eta)

    def grad_phi(self, u):
        """
        Gradient of the log posterior.

        Parameters
        ----------
        u: np.ndarray
            vector to evaluate the log-posterior at.
        """
        diff = self.A @ u - self.b
        resid = self.y_mean - self.H @ u
        return (self.A.T @ (self.G_inv @ diff) -
                self.n_obs * self.H.T @ (self.R_inv @ resid))

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
