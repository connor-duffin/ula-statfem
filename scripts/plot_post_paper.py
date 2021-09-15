import os
import gc
import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arviz import autocorr, ess
from tabulate import tabulate

from plot_samplers_2d import read_data


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


def traceplot_scalar(file_ids, labels, output_file):
    n_plots = len(file_ids)
    fig, axs = plt.subplots(1,
                            1,
                            constrained_layout=True,
                            figsize=(5, 3),
                            sharex=True)
    # axins = axs.inset_axes([0.45, 0.2, 0.5, 0.5])

    for i in range(n_plots):
        fem_dof, _ = read_data(SAMPLER_FILES[file_ids[i]], dofs=[100])
        axs.plot(fem_dof.T, alpha=0.8, label=labels[i])
        # axins.plot(fem_dof[:, 0:subset].T, alpha=0.8, label=labels[i])

    # axins.set_ylim(*axs.get_ylim())
    # axins.set_xlim(-50, 1000)
    # axins.set_xticks([0., 500])
    # axins.set_yscale("log")
    # axins.set_yticks([])
    # axins.set_xticklabels("")
    # axins.set_yticklabels("")

    axs.set_ylabel(r"$u^{(100)}$")
    axs.set_xlabel(r"Iteration $k$")

    fig.savefig(output_file, dpi=300)
    plt.close()


def traceplot_scalar_time(file_ids, labels, subset, output_file):
    """
    Plot the last `subset` samples from the chains, against wallclock time.
    """
    n_plots = len(file_ids)
    fig, axs = plt.subplots(1,
                            1,
                            constrained_layout=True,
                            figsize=(5, 3),
                            sharex=True)

    for i in range(n_plots):
        with h5py.File(SAMPLER_FILES[file_ids[i]], "r") as f:
            fem_dof = f["samples"][100, -subset:]
            n_sample = f["samples"].shape[1]
            t_sample = f.attrs["t_sample"]

        t = np.linspace(0, t_sample * (subset / (n_sample - 10_000)), subset)
        axs.plot(t[t <= 100], fem_dof.T[t <= 100], alpha=0.8, label=labels[i])

    axs.set_ylabel(r"$u^{(100)}$")
    axs.set_xlabel(r"Wallclock time (s)")

    fig.savefig(output_file, dpi=300)
    plt.close()


def acf_scalar(file_ids, labels, output_dir, warmup=0):
    """Plot ACF, traceplots, and print ESS of the scalar. """
    n_plots = len(file_ids)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))

    max_lags = 200
    for i in range(n_plots):
        log_measure, _ = read_data(SAMPLER_FILES[file_ids[i]],
                                   dofs=[100],
                                   warmup=warmup)

        log_measure = log_measure[0, :]
        n_samples = log_measure.shape[0]
        # ACF
        acfs = autocorr(log_measure)
        print(acfs.shape)
        ax.plot(acfs[:max_lags], ".-", label=f"{labels[i]}", markevery=10)

        # ESS
        n_eff = ess(log_measure)
        print(f"ESS ({labels[i]}): {n_eff}")

        # integrated ACF
        tau = n_samples / n_eff
        print(f"Integrated ACF: {tau:.6f}")

    ax.set_ylabel("ACF")
    ax.set_xlabel(r"Lag $j$")
    ax.legend()
    fig.savefig(output_dir + "acf-scalar.png", dpi=300)
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
        if label == "Exact":
            s, _ = read_data(SAMPLER_FILES[file_id], warmup=0)
            row.append("---")
            row.append("---")
        else:
            s, _ = read_data(SAMPLER_FILES[file_id], warmup=warmup)

            mean_approx[:] = np.mean(s, axis=1)
            var_approx[:] = np.var(s, axis=1)

            mean_error = norm(mean_exact - mean_approx) / norm_mean_exact
            var_error = norm(var_exact - var_approx) / norm_var_exact

            row.append(f"{mean_error:.6f}")
            row.append(f"{var_error:.6f}")

        if include_ess:
            if label == "Exact":
                log_measure, t_sample = read_data(SAMPLER_FILES[file_id],
                                                  dofs=ess_dof,
                                                  warmup=0)
            else:
                log_measure, t_sample = read_data(SAMPLER_FILES[file_id],
                                                  dofs=ess_dof,
                                                  warmup=warmup)

            # NOTE: this assumes t_sample is for the post-warmup iterations
            samples_n_eff = ess(log_measure)
            row.append(f"{samples_n_eff / t_sample:.3f}")

        table.append(row)
        row = []

        del s
        gc.collect()

    header_simple = ["sampler", "mean rel. error", "var rel. error"]
    header = [
        "Sampler", "$\\lVert \\mathbb{E}_K (u) - \\mathbb{E} (u) \\rVert /" +
        "\\lVert \\mathbb{E} (u) \\rVert$",
        "$\\lVert \\mathrm{var}_K (u) - \\mathrm{var} (u)\\rVert / " +
        "\\lVert \\mathrm{var}(u) \\rVert$"
    ]

    if include_ess:
        header_simple.append("ess / s")
        header.append("ESS / s")

    print(tabulate(table, headers=header_simple, numalign="right"))

    with open(output_file, "w") as f:
        f.write(
            tabulate(table,
                     headers=header,
                     numalign="right",
                     tablefmt="latex_raw"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nx", type=int)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n_warmup", type=int, default=1)
    args = parser.parse_args()

    SAMPLER_FILES = {
        "data": args.input_dir + "data.h5",
        "exact": args.input_dir + "exact.h5",
        "ula": args.input_dir + "ula.h5",
        "pula": args.input_dir + "pula.h5",
        "mala": args.input_dir + "mala.h5",
        "pmala": args.input_dir + "pmala.h5",
        "pcn": args.input_dir + "pcn.h5"
    }

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trace_dofs = np.linspace(100, (args.nx + 1)**2 - 1, 8, dtype=np.int64)

    file_ids = ["ula", "pula", "mala", "pmala", "pcn"]
    labels = ["ULA", "pULA", "MALA", "pMALA", "pCN"]

    plot_observations_mesh(args.nx, SAMPLER_FILES["data"],
                           output_dir + "mesh-obs-locations.png")
    traceplot_scalar_time(file_ids, labels, 5000,
                          output_dir + "traceplot-times.png")
    traceplot_dofs(trace_dofs, file_ids, labels,
                   output_dir + "traceplot-dofs.png")
    acf_scalar(file_ids, labels, output_dir, warmup=args.n_warmup)
    traceplot_scalar(file_ids, labels, output_dir + "traceplot-scalar.png")

    file_ids.insert(0, "exact")
    labels.insert(0, "Exact")

    sampler_errors_table(args.n_warmup,
                         file_ids,
                         labels,
                         output_dir + "error-table.tex",
                         include_ess=True,
                         ess_dof="log_measure")
