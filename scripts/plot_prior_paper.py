import gc
import os

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arviz import autocorr, ess
from tabulate import tabulate

from plot_samplers_2d import read_data


def rolling_mean(x):
    """Compute the rolling mean of a set of samples. """
    n = x.shape[0]
    xc = np.cumsum(x)
    ns = np.array(range(1, n + 1))
    mean = xc / ns
    return mean


def rel_diff(x_est, x):
    return np.abs(x - x_est) / np.abs(x)


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
            axs[i].set_xlabel(r"Iteration $k$")
            axs[i].set_ylabel(
                fr"$\Vert \mathbb{{ E }}_k(u^{{ ({dofs[i]}) }}) - \mathbb{{ E }}(u^{{ ({dofs[i]})}}) \Vert / \Vert \mathbb{{ E }}(u^{{ ({dofs[i]}) }}) \Vert $",
                fontsize="small")

    if n_dofs != 1:
        axs[0].legend()

    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_scalar(file_ids, labels, output_dir, warmup=0, dofs=[0]):
    """Plot ACF, traceplots, and print ESS of the scalar. """
    n_plots = len(file_ids)
    fig_trace, ax_trace = plt.subplots(1,
                                       1,
                                       constrained_layout=True,
                                       figsize=(4, 3))

    # assume that exact samples are in the first entry
    fig_acf, axs_acf = plt.subplots(1,
                                    1,
                                    constrained_layout=True,
                                    figsize=(4, 3))

    max_lags = 100
    lags = list(range(max_lags))
    acfs = np.zeros((max_lags - 1, ))
    for i in range(n_plots):
        scalar, _ = read_data(SAMPLER_FILES[file_ids[i]],
                              dofs=dofs,
                              warmup=warmup)
        scalar = scalar[0]

        # ESS
        samples_n_eff = ess(scalar)
        print(f"ESS ({labels[i]}): {samples_n_eff}")

        # traceplots
        ax_trace.plot(scalar, alpha=0.8, label=labels[i])

        # ACF
        acfs = autocorr(scalar)
        axs_acf.plot(lags, acfs[:max_lags], ".-", label=f"{labels[i]}")

    axs_acf.set_ylabel("ACF")
    axs_acf.set_xlabel(r"Lag $j$")
    axs_acf.legend()
    fig_acf.savefig(output_dir + "acf-scalar.png", dpi=300)

    ax_trace.set_ylabel(fr"$u^{{ ({dofs[0]}) }}$")
    ax_trace.set_xlabel(r"Iteration $k$")
    fig_trace.savefig(output_dir + "traceplot-scalar.png", dpi=300)

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

        row.append(f"{mean_error:.6f}")
        row.append(f"{var_error:.6f}")

        if include_ess:
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
        "Sampler",
        "$\\mathsf{Error}(\\mathbb{E}(u))$",
        "$\\mathsf{Error}(\\mathrm{var}(u))$"
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
    args = parser.parse_args()

    SAMPLER_FILES = {
        "data": args.input_dir + "data.h5",
        "exact": args.input_dir + "exact.h5",
        "pula": args.input_dir + "pula.h5",
        "mala": args.input_dir + "mala.h5",
        "pmala": args.input_dir + "pmala.h5"
    }

    os.makedirs(args.output_dir, exist_ok=True)

    trace_dofs = np.linspace(100, (args.nx + 1)**2 - 1, 8, dtype=np.int64)
    file_ids = ["pula", "pmala", "mala"]
    labels = ["pULA", "pMALA", "MALA"]

    plot_dof_convergence([100], file_ids, labels,
                         args.output_dir + "dof-convergence.png")
    plot_scalar(file_ids,
                labels,
                args.output_dir,
                warmup=args.n_warmup,
                dofs=[100])
    sampler_errors_table(args.n_warmup,
                         file_ids,
                         labels,
                         args.output_dir + "error-table.tex",
                         include_ess=True,
                         ess_dof=[100])
