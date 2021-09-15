import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from plot_samplers_2d import read_data


def plot_mean_convergence(file_ids, labels, output_dir):
    samples_exact, _ = read_data(SAMPLER_FILES["exact"], warmup=0)
    ns = np.array(list(range(1, samples_exact.shape[1] + 1)))

    norm = np.linalg.norm
    mean_exact = np.mean(samples_exact, axis=1)[:, np.newaxis]
    norm_mean_exact = norm(mean_exact)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))

    for fid, label in zip(file_ids, labels):
        samples, _ = read_data(SAMPLER_FILES[fid])
        rolling_mean = np.cumsum(samples, axis=1) / ns
        diff = rolling_mean - mean_exact

        if fid == "exact":
            # HACK: null plot to increment the color wheel
            ax.plot([], [], "-", label=label)
        else:
            ax.semilogy(norm(diff, axis=0) / norm_mean_exact, label=label)

    ax.set_ylabel(r"$\Vert \mathbb{E}_k(u) - \mathbb{E}(u) \Vert / \Vert \mathbb{E}(u) \Vert $")
    ax.set_xlabel(r"Iteration $k$")
    ax.legend(loc="upper right")
    plt.savefig(output_dir + "mean-error-convergence.png", dpi=300)
    plt.close()


def plot_log_measure(file_ids, labels, output_dir):
    """Traceplots of the log-measure. """
    n_plots = len(file_ids)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))

    for i in range(n_plots):
        log_measure, _ = read_data(SAMPLER_FILES[file_ids[i]],
                                   dofs="log_measure")

        # traceplots
        ax.plot(log_measure.T, label=labels[i])

    ax.set_ylabel(r"$\log p(u_k | \theta_k)$")
    ax.set_xlabel(r"Iteration $k$")
    ax.legend()
    fig.savefig(output_dir + "traceplot-log-measure.png", dpi=300)

    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nx", type=int)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    SAMPLER_FILES = {
        "exact": args.input_dir + "exact.h5",
        "pula": args.input_dir + "pula.h5",
        "pula_smallstep": args.input_dir + "pula-smallstep.h5",
    }

    file_ids = ["exact", "pula", "pula_smallstep"]
    labels = [
        "Exact", r"pULA, $\eta = 2 \times 10^{-1}$",
        r"pULA, $\eta = 1 \times 10^{-3}$"
    ]

    plot_mean_convergence(file_ids, labels, args.output_dir)
    plot_log_measure(file_ids, labels, args.output_dir)
