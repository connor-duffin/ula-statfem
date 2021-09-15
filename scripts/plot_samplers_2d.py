import h5py
import gc
import os

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arviz import ess, plot_autocorr
from tabulate import tabulate
from scipy.sparse.linalg import eigsh, LinearOperator

from run_samplers_2d import init_pois


def rolling_mean(x):
    """Compute the rolling mean of a set of samples. """
    n = x.shape[0]
    xc = np.cumsum(x)
    ns = np.array(range(1, n + 1))
    mean = xc / ns
    return mean


def rel_diff(x_est, x):
    return np.abs(x - x_est) / np.abs(x)


def rel_error(approx, exact, qoi=np.mean):
    norm = np.linalg.norm
    qoi_approx = qoi(approx, axis=0)
    qoi_exact = qoi(exact, axis=0)
    return norm(qoi_approx - qoi_exact) / norm(qoi_exact)


def read_data(filename, dofs=None, warmup=0):
    output = h5py.File(filename, "r")

    if type(dofs) == np.ndarray or type(dofs) == list:
        samples = output["samples"][dofs, warmup:]
    elif dofs is None:
        samples = output["samples"][:, warmup:]
    elif dofs == "log_measure":
        samples = output["log_measure"][warmup:]

    try:
        t_sample = output.attrs["t_sample"]
    except KeyError:
        t_sample = None

    try:
        t_setup = output.attrs["t_setup"]
        t_sample += t_setup
    except (KeyError, TypeError):
        pass

    return samples, t_sample


def read_eta(file_ids, labels):
    for file_id, label in zip(file_ids, labels):
        with h5py.File(SAMPLER_FILES[file_id], "r") as output:
            eta = output.attrs["eta"]

        print(f"Sampler {label}, eta = {eta:.8e}")


def plot_mean_var_fields(nx):
    pois = init_pois(nx)
    x1, x2 = pois.x_dofs[:, 0], pois.x_dofs[:, 1]

    samples_exact, _ = read_data(SAMPLER_FILES["exact"])
    samples_langevin, _ = read_data(SAMPLER_FILES["pula_lu_fixed"])

    mean_exact = np.mean(samples_exact, axis=1)
    mean_pula = np.mean(samples_langevin, axis=1)

    fig, axs = plt.subplots(2,
                            2,
                            sharex=True,
                            sharey=True,
                            constrained_layout=True,
                            figsize=(6, 4))
    axs = axs.flatten()
    im = axs[0].tricontourf(x1, x2, mean_exact, 64)
    plt.colorbar(im, ax=axs[0])
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_ylabel(r"$x_2$")
    axs[0].set_title("Est. mean: Exact")

    im = axs[1].tricontourf(x1, x2, mean_pula, 64)
    plt.colorbar(im, ax=axs[1])
    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_title("Est. mean: pULA (LU)")

    var_exact = np.var(samples_exact, axis=1)
    var_pula = np.var(samples_langevin, axis=1)

    im = axs[2].tricontourf(x1, x2, var_exact, 64)
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel(r"$x_1$")
    axs[2].set_ylabel(r"$x_2$")
    axs[2].set_title("Est. var: Exact")

    im = axs[3].tricontourf(x1, x2, var_pula, 64)
    plt.colorbar(im, ax=axs[3])
    axs[3].set_xlabel(r"$x_1$")
    axs[3].set_title("Est. var: pULA (LU)")
    plt.savefig(OUTPUT_DIR + "mean-var-samples.png", dpi=300)
    plt.close()


def plot_observations_mesh(nx, data_file, output_file):
    from run_samplers_2d_posterior import init_pois
    pois = init_pois(nx, data_file)

    mesh = pois.x_dofs
    x_obs = pois.x_obs

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(3, 2.5))
    ax.triplot(mesh[:, 0], mesh[:, 1], linewidth=0.5, label="FEM mesh")
    ax.plot(x_obs[:, 0],
            x_obs[:, 1],
            "o",
            color="tab:orange",
            label="Observation locations")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_covariance_conditioning(data_file, output_file):
    from run_samplers_2d_posterior import init_pois

    def compute_spectrum(nx):
        pois = init_pois(nx, data_file)

        pois.sample_theta()
        pois.assemble_A()

        C_inv = pois.compute_precision()

        def mv(x):
            return pois.factor.solve_A(x)

        C = LinearOperator(C_inv.shape, mv)
        vals = eigsh(C, k=100, return_eigenvectors=False)

        def mv_pc(x):
            w = pois.factor.solve_A(x)
            return pois.M @ w

        C_pc = LinearOperator(C_inv.shape, mv_pc)
        vals_pc = eigsh(C_pc, k=100, return_eigenvectors=False)
        return vals, vals_pc

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    axs = axs.flatten()
    nx_all = [16, 32, 64, 100]
    for n in nx_all:
        vals, vals_pc = compute_spectrum(n)
        axs[0].semilogy(vals[::-1], ".-", label=f"{(n + 1)**2} DOFs")
        axs[1].plot(vals_pc[::-1], ".-", label=f"{(n + 1)**2} DOFs")

    axs[0].legend()
    axs[0].set_title("No preconditioning")
    axs[1].set_title("Mean-theta preconditioning")

    axs[0].set_ylabel(r"$\lambda_j$")
    axs[1].set_ylabel(r"$\lambda_j$")

    axs[0].set_xlabel(r"$j$")
    axs[1].set_xlabel(r"$j$")
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_dof_convergence(dofs, file_ids, labels, output_file):
    """MC convergence plot for `dofs`. """
    n_dofs = len(dofs)

    if n_dofs == 1:
        n_cols = 1
        n_rows = 1
        figsize = (4, 3)
    else:
        n_cols = 4
        n_rows = n_dofs // n_cols
        figsize = (16, 3 * n_rows)

    samples_exact, _ = read_data(SAMPLER_FILES["exact"], dofs=dofs)
    fig, axs = plt.subplots(n_rows,
                            n_cols,
                            constrained_layout=True,
                            figsize=figsize)
    if n_dofs != 1:
        axs = axs.flatten()
    else:
        # HACK: in order for the loop to work with each
        axs = [axs]

    for file_id, label in zip(file_ids, labels):
        samples, _ = read_data(SAMPLER_FILES[file_id], dofs=dofs)
        for i in range(n_dofs):
            mean_exact_dof = np.mean(samples_exact[i, :])
            axs[i].loglog(rel_diff(rolling_mean(samples[i, :]),
                                   mean_exact_dof),
                          label=label)
            axs[i].set_title(f"DOF {dofs[i]}")

    if n_dofs != 1:
        axs[0].legend()

    plt.savefig(output_file, dpi=300)
    plt.close()


def traceplot_debug(filename, dofs):
    """Simplified traceplot for a single file. """
    alpha = 1.
    n_dofs = len(dofs)
    n_cols = 4
    n_rows = n_dofs // n_cols

    samples, _ = read_data(filename, dofs=dofs)
    fig, axs = plt.subplots(n_rows,
                            n_cols,
                            constrained_layout=True,
                            figsize=(15, 3 * n_rows))
    axs = axs.flatten()
    for i in range(n_dofs):
        axs[i].plot(samples[:, i], alpha=alpha)
        axs[i].set_title(f"DOF {dofs[i]}")

    fig.suptitle(f"Samples from {filename}")
    plt.savefig("figures/traceplot-debug.png", dpi=300)
    plt.close()


def traceplot_dofs(dofs, file_ids, labels, output_file):
    """Traceplot for `dofs` sampled through each method. """
    n_cols = 4
    n_dofs = len(dofs)
    n_rows = n_dofs // n_cols

    fig, axs = plt.subplots(n_rows,
                            n_cols,
                            constrained_layout=True,
                            figsize=(15, 3 * n_rows))
    axs = axs.flatten()
    for file_id, label in zip(file_ids, labels):
        samples, _ = read_data(SAMPLER_FILES[file_id], dofs=dofs)

        for i in range(n_dofs):
            axs[i].plot(samples[i, :], label=label)
            axs[i].set_title(f"DOF {dofs[i]}")

    axs[0].legend()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_scalar(file_ids, labels, output_dir, warmup=0, dofs="log_measure"):
    """Plot ACF, traceplots, and print ESS of the scalar. """

    def autocorr(x, t=1):
        """Compute the autocorrelation from a 1D array. """
        return np.corrcoef(np.array([x[:-t], x[t:]]))

    n_plots = len(file_ids)
    fig_trace, ax_trace = plt.subplots(1,
                                       1,
                                       constrained_layout=True,
                                       figsize=(4, 3))

    # assume that exact samples are in the first entry
    fig_acf, axs_acf = plt.subplots(1, 1,
                                    constrained_layout=True,
                                    figsize=(4, 3))

    zorder = 100
    max_lags = 100
    lags = list(range(1, max_lags))
    acfs = np.zeros((max_lags - 1, ))
    for i in range(n_plots):
        log_measure, _ = read_data(SAMPLER_FILES[file_ids[i]],
                                   dofs=dofs,
                                   warmup=warmup)

        # ensure the correct shape: (n_chains, n_samples)
        print(log_measure.shape)
        if log_measure.shape[0] != 1:
            log_measure = log_measure.T

        # ESS
        samples_n_eff = ess(log_measure)
        print(f"ESS ({labels[i]}): {samples_n_eff}")

        # traceplots
        ax_trace.plot(log_measure.T, alpha=0.8, label=labels[i], zorder=zorder)

        # ACF
        for j, lag in enumerate(lags):
            acfs[j] = autocorr(log_measure[0], lag)[0, 1]
        axs_acf.plot(lags, acfs, ".-", label=f"{labels[i]}", zorder=zorder)
        zorder -= 5

    axs_acf.set_ylabel("ACF")
    axs_acf.set_xlabel(r"Lag $j$")
    axs_acf.legend()
    fig_acf.savefig(output_dir + "acf-scalar.png", dpi=300)

    if dofs == "log_measure":
        ax_trace.set_title(r"$\log p(y | u_k)$")

    ax_trace.set_xlabel(r"Iteration $k$")
    fig_trace.savefig(output_dir + "traceplot-scalar.png", dpi=300)

    plt.close()


def ess_log_measure(file_ids, labels, warmup=0, dofs="log_measure"):
    for file_id, label in zip(file_ids, labels):
        log_measure, t = read_data(SAMPLER_FILES[file_id],
                                   dofs=dofs,
                                   warmup=warmup)

        samples_n_eff = ess(log_measure.T)
        print(f"ESS ({label}): {samples_n_eff}")


def plot_sampler_times(file_ids, labels, output_file):
    dofs = [0]  # dont care about variables, so choose arbitrary DOF

    times = []
    for file_id in file_ids:
        _, t = read_data(SAMPLER_FILES[file_id], dofs=dofs)
        times.append(t)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(labels, times)
    plt.xticks(rotation=45)
    ax.set_ylabel("Wallclock time (s)")
    plt.savefig(output_file, dpi=300)
    plt.close()


def sampler_errors_table(warmup,
                         file_ids,
                         labels,
                         output_file,
                         include_ess=False,
                         ess_dof="log_measure"):
    samples_exact, _ = read_data(SAMPLER_FILES["exact"], warmup=0)

    norm = np.linalg.norm
    mean_exact = np.mean(samples_exact, axis=1)
    norm_mean_exact = norm(mean_exact)

    var_exact = np.var(samples_exact, axis=1)
    norm_var_exact = norm(var_exact)

    mean_approx = np.zeros_like(samples_exact[:, 0])
    var_approx = np.zeros_like(samples_exact[:, 0])

    del samples_exact
    gc.collect()

    row = []
    table = []
    for label, file_id in zip(labels, file_ids):
        print(f"table row: {label}")
        row.append(label)
        s, _ = read_data(SAMPLER_FILES[file_id], warmup=warmup)

        mean_approx[:] = np.mean(s, axis=1)
        var_approx[:] = np.var(s, axis=1)

        mean_error = norm(mean_exact - mean_approx) / norm_mean_exact
        var_error = norm(var_exact - var_approx) / norm_var_exact

        row.append(mean_error)
        row.append(var_error)

        if include_ess:
            log_measure, t_sample = read_data(SAMPLER_FILES[file_id],
                                              dofs=ess_dof,
                                              warmup=warmup)
            # NOTE: this assumes t_sample is for the post-warmup iterations
            samples_n_eff = ess(log_measure)
            row.append(samples_n_eff / t_sample)

        table.append(row)
        row = []

        del s
        gc.collect()

    header_simple = ["sampler", "mean rel. error", "var rel. error"]
    header = [
        "Sampler", "$\\lVert \\mathbb{E}_N (u) - \\mathbb{E} (u) \\rVert /" +
        "\\lVert \\mathbb{E} (u) \\rVert$",
        "$\\lVert \\mathrm{var}_N (u) - \\mathrm{var} (u)\\rVert / " +
        "\\lVert \\mathrm{var}(u) \\rVert$"
    ]

    if include_ess:
        header_simple.append("ess / s")
        header.append("ESS / s")

    print(tabulate(table, headers=header_simple))

    with open(output_file, "w") as f:
        f.write(tabulate(table, headers=header, tablefmt="latex_raw"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nx", type=int)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--prior", action="store_true")
    args = parser.parse_args()

    SAMPLER_FILES = {
        "data": args.input_dir + "data.h5",
        "exact": args.input_dir + "exact.h5",
        "ula": args.input_dir + "ula.h5",
        "pula": args.input_dir + "pula.h5",
        "mala": args.input_dir + "mala.h5",
        "pmala": args.input_dir + "pmala.h5",
        "pcn": args.input_dir + "pcn.h5",
        "pula_smallstep": args.input_dir + "pula-smallstep.h5",
        "pula_exact": args.input_dir + "pula-exact.h5",
        "pula_lu": args.input_dir + "pula-lu.h5",
        "pula_lu_fixed": args.input_dir + "pula-lu-fixed.h5"
    }

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trace_dofs = np.linspace(100, (args.nx + 1)**2 - 1, 8, dtype=np.int64)

    if args.prior:
        file_ids = ["pula", "pmala", "mala"]
        labels = ["pULA", "pMALA", "MALA"]
    else:
        file_ids = ["ula", "pula", "mala", "pmala", "pcn"]
        labels = ["ULA", "pULA", "MALA", "pMALA", "pCN"]

    if args.prior:
        plot_dof_convergence([100], file_ids, labels,
                             OUTPUT_DIR + "dof-convergence.png")

    file_ids.append("exact")
    labels.append("Exact")

    traceplot_dofs(trace_dofs, file_ids, labels, OUTPUT_DIR + "traceplot.png")
    plot_sampler_times(file_ids, labels, OUTPUT_DIR + "sampler-times.png")

    if args.prior:
        plot_scalar(file_ids,
                    labels,
                    OUTPUT_DIR,
                    warmup=args.n_warmup,
                    dofs=[100])
        sampler_errors_table(args.n_warmup,
                             file_ids,
                             labels,
                             OUTPUT_DIR + "error-table.tex",
                             include_ess=True,
                             ess_dof="log_measure")
    else:
        # plot_scalar(file_ids,
        #             labels,
        #             OUTPUT_DIR,
        #             warmup=args.n_warmup,
        #             dofs="log_measure")
        sampler_errors_table(args.n_warmup,
                             file_ids,
                             labels,
                             OUTPUT_DIR + "error-table.tex",
                             include_ess=True,
                             ess_dof="log_measure")

    # if not args.prior:
    #     plot_observations_mesh(args.nx,
    #                            SAMPLER_FILES["data"],
    #                            output_file=OUTPUT_DIR + "mesh-obs.png")

    #     plot_covariance_conditioning(SAMPLER_FILES["data"],
    #                                  OUTPUT_DIR + "spectrum-decay.png")
    # else:
    #    plot_mean_var_fields(args.nx)

