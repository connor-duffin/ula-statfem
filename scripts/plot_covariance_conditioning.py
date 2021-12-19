from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from run_samplers_2d import init_pois

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def compute_spectrum(nx, i):
    pois = init_pois(nx)

    pois.setup_pc("lu")
    A_mean = pois.A.copy()
    G_sqrt = pois.G_sqrt.tocsr()

    pois.sample_theta()
    pois.assemble_A()

    C_inv = pois.A.T @ pois.G_inv @ pois.A
    C_inv = C_inv.todense()

    A_factor = splu(A_mean.tocsc())
    X = A_factor.solve(G_sqrt @ C_inv).T
    C_inv_pc = A_factor.solve(G_sqrt @ X)

    vals = eigh(C_inv, eigvals_only=True)
    vals_pc = eigh(C_inv_pc, eigvals_only=True)

    kappa = vals[-1] / vals[0]
    kappa_pc = vals_pc[-1] / vals_pc[0]
    return kappa, kappa_pc


if __name__ == "__main__":
    data_file = "outputs/posterior-mesh-ll-32/data.h5"
    output_file = "figures/precision-conditioning.png"

    nx_all = [15, 20, 25, 30, 35]
    h = np.array([1 / nx for nx in nx_all])

    p = Pool(17)
    means = np.zeros_like(h)
    sds = np.zeros_like(h)

    means_pc = np.zeros_like(h)
    sds_pc = np.zeros_like(h)

    n_mc = 1000
    for i, n in enumerate(nx_all):
        print("On meshsize: ", n)
        out = p.starmap(compute_spectrum, [(n, i) for i in range(n_mc)])
        out = np.array(out)

        m = np.mean(out, axis=0)
        means[i] = m[0]
        means_pc[i] = m[1]

        sd = np.std(out, axis=0)
        sds[i] = sd[0]
        sds_pc[i] = sd[1]

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    axs = axs.flatten()
    axs[0].errorbar(h, means, 0.67 * sds, fmt=".-")
    axs[1].errorbar(h, means_pc, 0.67 * sds_pc, fmt=".-")

    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$h$")
    axs[0].set_ylabel(r"$\mathbb{E}_\theta[\kappa_\theta]$")
    axs[0].set_title("No preconditioning")

    axs[1].set_xlabel(r"$h$")
    axs[1].set_ylabel(r"$\mathbb{E}_\theta[\kappa_\theta^M]$")
    axs[1].set_title(r"Mean-$\theta$ preconditioning")
    plt.savefig(output_file, dpi=300)
    plt.close()
