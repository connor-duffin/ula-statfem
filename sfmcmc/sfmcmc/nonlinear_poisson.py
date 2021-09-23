import logging

import numpy as np
from dolfin import *

from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .utils import dolfin_to_csr, sq_exp_covariance, SquareExpKronecker

logger = logging.getLogger(__name__)


class NonlinearPoisson1D:
    def __init__(self, nx):
        self.mesh = UnitIntervalMesh(nx)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        v = TestFunction(self.V)
        self.u = Function(self.V)
        self.bc = DirichletBC(self.V, Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(self.bc.get_boundary_values().keys()))

        self.xi = Function(self.V)
        self.f = interpolate(
            Expression("8 * sin(pi * x[0])", degree=4), self.V)

        x0 = self.x_dofs[:, 0]
        self.F = ((1 + self.u**2) * inner(grad(self.u), grad(v)) -
                  inner(self.f, v) - inner(self.xi, v)) * dx
        self.J = derivative(self.F, self.u, TrialFunction(self.V))

        L = inner(self.f, v) * dx
        self.b = assemble(L)
        self.bc.apply(self.b)

        problem = NonlinearVariationalProblem(self.F,
                                              self.u,
                                              bcs=self.bc,
                                              J=self.J)
        self.solver = NonlinearVariationalSolver(problem)
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

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = inner(grad(u), grad(v)) * dx
        L = inner(self.f, v) * dx

        A, b = assemble(a), assemble(L)
        self.bc.apply(A, b)
        A, b = dolfin_to_csr(A), b[:]
        A_lu = splu(A.tocsc())

        xi = self.G_sqrt @ z
        xi[self.bc_dofs] = 0.
        u_scipy = A_lu.solve(b + xi)

        xi = Function(self.V)
        xi.vector()[:] = np.copy(self.M_lu.solve(self.G_sqrt @ z))
        L += inner(xi, v) * dx

        u = Function(self.V)
        solve(a == L, u, bcs=self.bc)
        u_fenics = u.vector()[:]

        norm = np.linalg.norm
        rel_diff = norm(u_scipy - u_fenics) / norm(u_fenics)
        assert rel_diff < rtol
        logger.info("rel diff: %e", rel_diff)

    def setup_G(self, sigma):
        """Set up the statFEM G matrix. """
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        m = inner(u, v) * dx
        M = as_backend_type(assemble(m))
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

    def assemble_system(self, u):
        self.u.vector()[:] = u
        J, F = assemble_system(self.J, self.F, bcs=self.bc)
        return F[:], dolfin_to_csr(J)

    def compute_trace_derivative(self, J_factor):
        # TODO: clean up this loop
        tr = np.zeros((self.n_dofs, ))
        for i in range(self.n_dofs):
            d = Function(self.V)
            d.vector()[i] = 1.

            H = assemble(derivative(self.J, self.u, d))
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
        log_det = np.sum(np.log(J_lu.L.diagonal())
                         + np.log(J_lu.U.diagonal()))
        return -log_det + F.T @ self.G_inv @ F / 2

    def ula_step(self, eta=1e-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        u_curr -= eta * self.grad_phi(u_curr) + np.sqrt(2 * eta) * z
        self.u_curr[:] = u_curr

    def tula_step(self, eta=1-2):
        z = np.random.normal(size=(self.n_dofs, ))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        grad_phi = self.grad_phi(u_curr)
        gradnorm = np.linalg.norm(grad_phi)
        u_curr -= eta * (1 / (1 + eta * gradnorm)) * grad_phi + np.sqrt(2*eta) * z
        self.u_curr[:] = u_curr

    def tulac_step(self, eta=1 - 2):
        z = np.random.normal(size=(self.n_dofs,))
        z[self.bc_dofs] = 0.

        u_curr = self.u_curr
        grad_phi = self.grad_phi(u_curr)
        gradnorm = np.abs(grad_phi)
        u_curr -= eta * (1 / (1 + eta * gradnorm)) * grad_phi + np.sqrt(2 * eta) * z
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
        solve(self.F == 0, self.u, bcs=self.bc, J=self.J)
        mean = np.copy(self.u.vector()[:])

        J = assemble(self.J)
        self.bc.apply(J)
        J_scipy = dolfin_to_csr(J)
        J_scipy_lu = splu(J_scipy.tocsc())

        temp = J_scipy_lu.solve(self.G.todense())
        cov = J_scipy_lu.solve(temp.T)

        return mean, cov


class NonlinearPoisson:
    def __init__(self, nx):
        self.mesh = UnitSquareMesh(nx, nx)

        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.x_dofs = self.V.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        v = TestFunction(self.V)
        self.u = Function(self.V)
        self.bc = DirichletBC(self.V, Constant(0), "on_boundary")
        self.bc_dofs = np.array(list(self.bc.get_boundary_values().keys()))

        self.xi = Function(self.V)
        self.f = interpolate(
            Expression("sin(pi * x[0]) * cos(pi * x[1])", degree=4), self.V)

        self.theta = Function(self.V)
        x0, x1 = self.x_dofs[:, 0], self.x_dofs[:, 1]
        self.theta_mean = 1 + 0.3 * np.sin(np.pi * (x0 + x1))

        self.F = ((self.theta + self.u**2) * inner(grad(self.u), grad(v)) -
                  inner(self.f, v) - inner(self.xi, v)) * dx
        self.J = derivative(self.F, self.u, TrialFunction(self.V))

        L = inner(self.f, v) * dx
        self.b = assemble(L)
        self.bc.apply(self.b)

        problem = NonlinearVariationalProblem(self.F,
                                              self.u,
                                              bcs=self.bc,
                                              J=self.J)
        self.solver = NonlinearVariationalSolver(problem)
        prm = self.solver.parameters
        prm["newton_solver"]["linear_solver"] = "gmres"
        prm["newton_solver"]["preconditioner"] = "petsc_amg"

        self.u_curr = np.zeros((self.n_dofs, ))
        self.J_PC_lu = None

    def setup_theta(self, scale, ell, method="default", nugget=1e-10):
        """Setup the GP diffusivity theta. """
        logger.info("starting theta setup")
        self.theta_method = method

        if method == "default":
            x0, x1 = self.x_dofs[:, 0], self.x_dofs[:, 1]
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

    def generate_theta(self):
        """Sample theta into the Fenics function self.theta. """
        w = np.zeros_like(self.theta_mean)

        if self.theta_method == "default":
            z = np.random.normal(size=(self.n_dofs, ))
            w[:] = self.K_theta_chol @ z
        else:
            w[:] = self.theta_gp.sample()

        self.theta.vector()[:] = np.copy(self.theta_mean + w)

    def linear_solve_xi_test(self, rtol=1e-10):
        """Test that Scipy and Fenics solve align to relative tolerance rtol.

        Verifies that premultiplying by M^{-1} does in fact scale as needed.
        """
        z = self.generate_z()

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = inner(grad(u), grad(v)) * dx
        L = inner(self.f, v) * dx

        A, b = assemble(a), assemble(L)
        self.bc.apply(A, b)
        A, b = dolfin_to_csr(A), b[:]
        A_lu = splu(A.tocsc())

        xi = self.G_sqrt @ z
        xi[self.bc_dofs] = 0.
        u_scipy = A_lu.solve(b + xi)

        xi = Function(self.V)
        xi.vector()[:] = np.copy(self.M_lu.solve(self.G_sqrt @ z))
        L += inner(xi, v) * dx

        u = Function(self.V)
        solve(a == L, u, bcs=self.bc)
        u_fenics = u.vector()[:]

        norm = np.linalg.norm
        rel_diff = norm(u_scipy - u_fenics) / norm(u_fenics)
        assert rel_diff < rtol
        logger.info("rel diff: %e", rel_diff)

    def setup_G(self, sigma):
        """Set up the statFEM G matrix. """
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        m = inner(u, v) * dx
        M = as_backend_type(assemble(m))
        M_scipy = dolfin_to_csr(M)
        self.M_lu = splu(M_scipy.tocsc())

        G_diag = sigma**2 * M.mat().getRowSum().array
        G_diag_inv = 1 / G_diag
        G_diag_sqrt = np.sqrt(G_diag)

        self.G = diags(G_diag)
        self.G_inv = diags(G_diag_inv)
        self.G_sqrt = diags(G_diag_sqrt)

    def setup_pc(self, pc_type):
        if pc_type == "lu":
            self.u.vector()[:] = 1.
            self.xi.vector()[:] = 0.
            self.theta.vector()[:] = self.theta_mean

            self.solver.solve()
            self.u_fem = np.copy(self.u.vector()[:])
            self.u_curr[:] = self.u.vector()[:]

            J, _ = assemble_system(self.J, self.F, bcs=self.bc)
            self.J_PC = dolfin_to_csr(J)
            self.J_PC_lu = splu(self.J_PC.tocsc())
        elif pc_type == "low_rank":
            pass
        else:
            raise ValueError(f"Preconditioner {pc_type} not supported")

    def assemble_system(self, u):
        self.u.vector()[:] = u
        J, F = assemble_system(self.J, self.F, bcs=self.bc)
        return F[:], dolfin_to_csr(J)

    def log_prop_mala(self, u_eval, u, F, J, eta):
        mean = u - eta * J.T @ (self.G_inv @ F)
        diff = u_eval - mean
        return -np.dot(diff, diff) / (4 * eta)

    def log_prop_pmala(self, u_eval, u, F, J, eta):
        J_lu = self.J_PC_lu

        grad_phi = J.T @ (self.G_inv @ F)
        drift = -eta * self.G @ (J_lu.solve(grad_phi, trans="T"))
        mean = u + J_lu.solve(drift)
        diff = u_eval - mean

        return -np.dot(self.J_PC @ diff, self.G_inv @ (self.J_PC @ diff)) / (
            4 * eta)

    def ula_step(self, eta=1e-2):
        """Complete a single ULA step. """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        F, J = self.assemble_system(self.u_curr)

        u_next = (self.u_curr - eta * J.T @ (self.G_inv @ F) -
                  np.sqrt(2 * eta) * z)
        self.u_curr[:] = u_next

    def mala_step(self, eta=1e-2):
        """Complete a single MALA step. """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        F_curr, J_curr = self.assemble_system(self.u_curr)
        drift = J_curr.T @ (self.G_inv @ F_curr)
        diff = np.sqrt(2 * eta) * z
        update = -eta * drift + diff
        update[self.bc_dofs] = 0.  # sanity check

        u_prop = (self.u_curr + update)
        F_prop, J_prop = self.assemble_system(u_prop)

        log_target_curr = -np.dot(F_curr, self.G_inv @ F_curr) / 2
        log_target_prop = -np.dot(F_prop, self.G_inv @ F_prop) / 2

        log_prop = self.log_prop_mala(u_prop, self.u_curr, F_curr, J_curr, eta)
        log_prop_flip = self.log_prop_mala(self.u_curr, u_prop, F_prop, J_prop,
                                           eta)

        log_alpha = np.min(
            [0, log_target_prop - log_target_curr + log_prop_flip - log_prop])

        if np.log(np.random.uniform()) <= log_alpha:
            self.u_curr[:] = u_prop
            accepted = True
        else:
            accepted = False

        return accepted

    def pmala_step(self, eta=1e-2):
        """Complete a single preconditioned-MALA step.

        Preconditioned with the mean-theta Hessian.
        """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        self.xi.vector()[:] = 0.
        F_curr, J_curr = self.assemble_system(self.u_curr)
        J_lu = self.J_PC_lu

        grad_phi = J_curr.T @ (self.G_inv @ F_curr)
        drift = -eta * self.G @ (J_lu.solve(grad_phi, trans="T"))
        diff = np.sqrt(2 * eta) * self.G_sqrt @ z
        update = J_lu.solve(drift + diff)
        u_prop = self.u_curr + update

        F_prop, J_prop = self.assemble_system(u_prop)

        log_target_curr = -np.dot(F_curr, self.G_inv @ F_curr) / 2
        log_target_prop = -np.dot(F_prop, self.G_inv @ F_prop) / 2

        log_prop = self.log_prop_pmala(u_prop, self.u_curr, F_curr, J_curr,
                                       eta)
        log_prop_flip = self.log_prop_pmala(self.u_curr, u_prop, F_prop,
                                            J_prop, eta)

        log_alpha = np.min(
            [0, log_target_prop - log_target_curr + log_prop_flip - log_prop])

        if np.log(np.random.uniform()) <= log_alpha:
            self.u_curr[:] = u_prop
            accepted = True
        else:
            accepted = False

        return accepted

    def pula_step_exact(self, eta=1e-2):
        """Complete a preconditioned-ULA step.

        Preconditioned with the exact Hessian.
        """
        z = self.generate_z()
        z[self.bc_dofs] = 0.

        self.xi.vector()[:] = 0.
        F, J = self.assemble_system(self.u_curr)
        J_lu = splu(J.tocsc())

        grad_phi = J.T @ (self.G_inv @ F)
        drift = -eta * self.G @ (J_lu.solve(grad_phi, trans="T"))
        diff = np.sqrt(2 * eta) * self.G_sqrt @ z
        update = J_lu.solve(drift + diff)

        u_next = self.u_curr + update

        self.u_curr[:] = u_next

    def pula_step_lu(self, eta=1e-2):
        """Complete a preconditioned-ULA step.

        Preconditioned with the mean-theta Gauss-Newton hessian.
        """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        self.xi.vector()[:] = 0.
        F, J = self.assemble_system(self.u_curr)
        J_lu = self.J_PC_lu

        grad_phi = J.T @ (self.G_inv @ F)

        drift = -eta * self.G @ (J_lu.solve(grad_phi, trans="T"))
        diff = np.sqrt(2 * eta) * self.G_sqrt @ z
        update = J_lu.solve(drift + diff)

        u_next = self.u_curr + update
        self.u_curr[:] = u_next

    def approx_sample(self):
        """Sample from first-order approximate measure with ULA. """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        self.xi.vector()[:] = 0.
        F, J = self.assemble_system(self.u_fem)
        J_lu = splu(J.tocsc())

        w = J_lu.solve(self.G_sqrt @ z)
        return self.u_fem + w

    def metropolis_step(self, eta=1e-2):
        """Sample from approximate measure using a symmetric proposal. """
        self.generate_theta()

        z = self.generate_z()
        z[self.bc_dofs] = 0.

        J_lu = self.J_PC_lu
        u_prop = self.u_curr + eta * J_lu.solve(self.G_sqrt @ z)

        F_curr, J_curr = self.assemble_system(self.u_curr)
        F_prop, J_prop = self.assemble_system(u_prop)

        log_target_curr = -np.dot(F_curr, self.G_inv @ F_curr) / 2
        log_target_prop = -np.dot(F_prop, self.G_inv @ F_prop) / 2

        log_alpha = np.min([0, log_target_prop - log_target_curr])

        if np.log(np.random.uniform()) <= log_alpha:
            self.u_curr[:] = u_prop
            accepted = True
        else:
            accepted = False

        return accepted

    def exact_sample(self):
        """Generate exact sample with Fenics. """
        self.generate_theta()
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
        solve(self.F == 0, self.u, bcs=self.bc, J=self.J)
        mean = np.copy(self.u.vector()[:])

        J = assemble(self.J)
        self.bc.apply(J)
        J_scipy = dolfin_to_csr(J)
        J_scipy_lu = splu(J_scipy.tocsc())

        temp = J_scipy_lu.solve(self.G.todense())
        cov = J_scipy_lu.solve(temp.T)

        return mean, cov
