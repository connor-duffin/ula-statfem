import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from arviz import ess
from run_samplers_2d_posterior import init_pois

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def estimate_tau(nx, eta, n_sample, n_inner, n_chains=1):
    lp = np.zeros((n_chains, n_sample))
    pois = init_pois(nx, data_file)

    for i in range(n_chains):
        pois.u[:] = pois.sample_posterior_exact()
        for j in range(n_sample):
            pois.pula_step(eta, fixed_theta=False)

            for k in range(n_inner):
                pois.pula_step(eta, fixed_theta=True)

            lp[i, j] = pois.log_likelihood(pois.u)

    n_eff = ess(lp)
    tau = (n_chains * n_sample) / n_eff
    print(f"Mesh {nx} finished, tau = {tau:.6e}")
    return tau


def estimate_var_error(nx, eta, n_sample, n_inner):
    pois = init_pois(nx, data_file)

    samples = np.zeros((pois.n_dofs, n_sample))
    for i in range(n_sample):
        pois.pula_step(eta, fixed_theta=False)

        for j in range(n_inner):
            pois.pula_step(eta, fixed_theta=True)

        samples[:, i] = pois.u

    samples_exact = np.zeros((pois.n_dofs, n_sample))
    for i in range(n_sample):
        samples_exact[:, i] = pois.sample_posterior_exact()

    print(f"Mesh {nx} finished")
    norm = np.linalg.norm
    var_approx = np.var(samples, axis=1)
    assert var_approx.shape == (pois.n_dofs, )

    var = np.var(samples_exact, axis=1)
    assert var.shape == (pois.n_dofs, )

    rel_error = norm(var - var_approx) / norm(var)
    return rel_error


nx = 32
n_sample = 2000
data_file = f"outputs/posterior-mesh-ll-{nx}/data.h5"

print("Number of CPUs: ", os.cpu_count())

# run samplers in parallel, for different n_inner values
print("Running for different inner iterations")
n_inners = list(range(1, 30, 1))
args = [(nx, (nx + 1)**(-2 / 3), n_sample, n, 1) for n in n_inners]

p = multiprocessing.Pool(17)
taus_n_inner = p.starmap(estimate_tau, args)

print("Running for different stepsize")
etas = np.linspace(1e-2, 1e-1, 20)
args = [(nx, eta, n_sample, 10, 1) for eta in etas]

p = multiprocessing.Pool(17)
taus_eta = p.starmap(estimate_tau, args)

print("Running for different meshsize")
nx = list(range(10, 65, 5))
h = [1 / n for n in nx]
args = [(n, (n + 1)**(-2/3), 10_000, 10) for n in nx]

p = multiprocessing.Pool(17)
error_h = p.starmap(estimate_var_error, args)

fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3))
axs = axs.flatten()
axs[0].plot(n_inners, taus_n_inner, ".-")
axs[0].set_xlabel(r"$n_{\mathrm{inner}}$")
axs[0].set_ylabel(r"$\tau$")

axs[1].plot(etas, taus_eta, ".-")
axs[1].set_xlabel(r"$\eta$")
axs[1].set_ylabel(r"$\tau$")

axs[2].plot(h, error_h, ".-")
axs[2].set_xlabel(r"$h$")
axs[2].set_ylabel(r"$\Vert \mathrm{var}(u_{ULA}) - \mathrm{var}(u) \Vert / \Vert \mathrm{var}(u) \Vert  $")
plt.savefig("figures/parameter-effects.pdf", dpi=400)
plt.show()
plt.close()
