import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from arviz import ess
from run_samplers_2d_posterior import init_pois

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def estimate_tau(nx, eta, n_sample, n_inner, n_chains=1, sampler="pula"):
    lp = np.zeros((n_chains, n_sample))
    pois = init_pois(nx, data_file, start="warm")

    if sampler == "pula":
        step = pois.pula_step
    elif sampler == "pmala":
        step = pois.pmala_step
    else:
        print("Given sampler not recognised")

    for i in range(n_chains):
        pois.u[:] = pois.sample_posterior_exact()
        for j in range(n_sample):
            step(eta, fixed_theta=False)

            for k in range(n_inner):
                step(eta, fixed_theta=True)

            lp[i, j] = pois.log_likelihood(pois.u)

    n_eff = ess(lp)
    tau = (n_chains * n_sample) / n_eff
    print(f"Mesh {nx} finished, tau = {tau:.6e}")
    return tau


def estimate_var_error(nx, eta, n_sample, n_inner, sampler):
    pois = init_pois(nx, data_file, start="warm")

    if sampler == "pula":
        step = pois.pula_step
    elif sampler == "pmala":
        step = pois.pmala_step
    else:
        print("Given sampler not recognised")

    samples = np.zeros((pois.n_dofs, n_sample))
    for i in range(n_sample):
        step(eta, fixed_theta=False)

        for j in range(n_inner):
            step(eta, fixed_theta=True)

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
n_inners = list(range(1, 20, 1))

pula_args = [(nx, (nx + 1)**(-2 / 3), n_sample, n, 1, "pula")
             for n in n_inners]
p = multiprocessing.Pool(17)
pula_taus_n_inner = p.starmap(estimate_tau, pula_args)

pmala_args = [(nx, (nx + 1)**(-2 / 3), n_sample, n, 1, "pmala")
              for n in n_inners]
pmala_taus_n_inner = p.starmap(estimate_tau, pmala_args)

print("Running for different stepsize")
etas = np.linspace(1e-2, 1e-1, 20)
pula_args = [(nx, eta, n_sample, 10, 1, "pula") for eta in etas]
pmala_args = [(nx, eta, n_sample, 10, 1, "pmala") for eta in etas]

p = multiprocessing.Pool(17)
pula_taus_eta = p.starmap(estimate_tau, pula_args)
pmala_taus_eta = p.starmap(estimate_tau, pmala_args)

print("Running for different meshsize")
n_sample_mesh = 10_000
nx = list(range(10, 65, 5))
h = [1 / n for n in nx]
pula_args = [(n, (n + 1)**(-2/3), n_sample_mesh, 10, "pula") for n in nx]
pmala_args = [(n, (n + 1)**(-2/3), n_sample_mesh, 10, "pmala") for n in nx]

p = multiprocessing.Pool(17)
pula_error_h = p.starmap(estimate_var_error, pula_args)
pmala_error_h = p.starmap(estimate_var_error, pmala_args)

fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3))
axs = axs.flatten()
axs[0].plot(n_inners, pula_taus_n_inner, ".-", label="pULA")
axs[0].plot(n_inners, pmala_taus_n_inner, ".-", label="pMALA")
axs[0].set_xlabel(r"$n_{\mathrm{inner}}$")
axs[0].set_ylabel(r"$\tau$")
axs[0].legend()

axs[1].plot(etas, pula_taus_eta, ".-")
axs[1].plot(etas, pmala_taus_eta, ".-")
axs[1].set_xlabel(r"$\eta$")
axs[1].set_ylabel(r"$\tau$")

axs[2].plot(h, pula_error_h, ".-")
axs[2].plot(h, pmala_error_h, ".-")
axs[2].set_xlabel(r"$h$")
axs[2].set_ylabel(r"$\mathsf{Error}(\mathrm{var}(u))$")
plt.savefig("figures/parameter-effects.pdf", dpi=400)
plt.show()
plt.close()
