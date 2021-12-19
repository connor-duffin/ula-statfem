import os
import h5py
import time
import logging

import numpy as np

from argparse import ArgumentParser

from sfmcmc.samplers import PoissonUnitTheta

logger = logging.getLogger(__name__)


def scale_eta(eta, acceptance_ratio, learning_rate, lower, upper):
    """Helper function to adapt the stepsize for MALA.

    If eta ∉ (1 - threshold, threshold) then scale by learning_rate.
    """
    if acceptance_ratio >= upper:
        eta *= (1 + learning_rate)
    elif acceptance_ratio <= lower:
        eta *= (1 - learning_rate)

    return eta


def init_pois(nx, start="warm"):
    pois = PoissonUnitTheta(nx)
    pois.setup_G(scale=0.05)
    pois.setup_theta(0.2, 0.2, method="kronecker")

    if start == "warm":
        pois.u[:] = pois.exact_step(method="amg")
    elif start == "cold":
        pois.u[:] = 0.

    return pois


def init_sample_array(n_sample, thin, k):
    """
    Create numpy array of proper shape to store samples.

    Returns
    -------
    samples : ndarray
        Array of zeros of shape (k, n_sample_thinned)
    """
    n_sample_thinned = n_sample // thin
    if n_sample % thin >= 1:
        n_sample_thinned += 1

    if k == 1:
        return np.zeros((n_sample_thinned, ))
    else:
        return np.zeros((k, n_sample_thinned))


def run_mala(nx,
             eta,
             output_file,
             n_sample,
             n_inner=1,
             n_warmup=1,
             start="warm"):
    prefix = "(mala)"
    pois = init_pois(nx, start=start)
    samples = init_sample_array(n_sample - n_warmup, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample - n_warmup, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    i_save = 0
    n_accept = 0
    t_start = time.time()
    print("sanity check")
    for i in range(n_sample):
        accepted = pois.mala_step(eta=eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.mala_step(eta=eta, fixed_theta=True)

        if accepted:
            n_accept += 1

        if i % 100 == 0 and i > 0:
            acceptance_ratio = n_accept / 100
            logger.info(prefix + f" iter: {i} / {n_sample}" +
                        f" last 100 iter. acc ratio: {acceptance_ratio:.2f}" +
                        f" eta = {eta:.6e}")

            # training gimmick
            if i < n_warmup:
                eta = scale_eta(eta,
                                acceptance_ratio,
                                learning_rate=0.1,
                                lower=0.4,
                                upper=0.6)

            n_accept = 0

        # don't store warmup iterations
        if i >= n_warmup:
            samples[:, i_save] = pois.u
            log_measure[i_save] = pois.log_target(pois.u)
            i_save += 1

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output = h5py.File(output_file, "w")
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta
        output.attrs["t_sample"] = t_sample
        output.attrs["n_warmup"] = n_warmup
        output.attrs["n_sample"] = n_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)


def run_pmala(nx,
              eta,
              output_file,
              n_sample,
              n_inner=1,
              n_warmup=1,
              start="warm"):
    prefix = "(pmala)"
    pois = init_pois(nx, start=start)
    samples = init_sample_array(n_sample - n_warmup, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample - n_warmup, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    i_save = 0
    n_accept = 0
    t_start = time.time()
    for i in range(n_sample):
        accepted = pois.pmala_step(eta=eta)

        for j in range(n_inner - 1):
            accepted = pois.pmala_step(eta=eta, fixed_theta=True)

        if accepted:
            n_accept += 1

        if i % 100 == 0:
            acceptance_ratio = n_accept / 100
            logger.info(prefix + f" iter: {i} / {n_sample}" +
                        f" last 100 iter. acc ratio: {acceptance_ratio:.2f}" +
                        f" eta = {eta:.6e}")

            # training gimmick
            if i < n_warmup:
                eta = scale_eta(eta,
                                acceptance_ratio,
                                learning_rate=0.1,
                                lower=0.4,
                                upper=0.6)

            n_accept = 0

        # don't store warmup iterations
        if i >= n_warmup:
            samples[:, i_save] = pois.u
            log_measure[i_save] = pois.log_target(pois.u)
            i_save += 1

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output = h5py.File(output_file, "w")
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta
        output.attrs["t_sample"] = t_sample
        output.attrs["n_warmup"] = n_warmup
        output.attrs["n_sample"] = n_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)


def run_pula_lu(nx,
                eta,
                output_file,
                n_sample,
                n_inner=1,
                n_warmup=0,
                start="warm"):
    prefix = "(pula_lu_mean)"
    pois = init_pois(nx, start=start)

    samples = init_sample_array(n_sample - n_warmup, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample - n_warmup, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    t_start = time.time()
    pois.setup_pc("lu")
    t_end_setup = time.time()
    t_setup = t_end_setup - t_start
    logger.info(prefix + f" Took {t_setup:.6f} s to compute pc")

    i_save = 0
    for i in range(n_sample):
        pois.pula_step_lu_mean(eta=eta, fixed_theta=False)

        for j in range(n_inner - 1):
            pois.pula_step_lu_mean(eta=eta, fixed_theta=True)

        if i >= n_warmup:
            samples[:, i_save] = pois.u
            log_measure[i_save] = pois.log_target(pois.u)
            i_save += 1

        if i % 10 == 0:
            logger.info(prefix + f" iter {i} / {n_sample}")

    t_end = time.time()
    t_sample = t_end - t_end_setup
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output = h5py.File(output_file, "w")
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta
        output.attrs["t_setup"] = t_setup
        output.attrs["t_sample"] = t_sample
        output.attrs["n_sample"] = n_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)


def run_exact(nx, output_file, n_sample):
    prefix = "(exact)"
    pois = init_pois(nx)
    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    t_start = time.time()
    for i in range(n_sample):
        u = pois.exact_step(method="amg")

        if i % 100 == 0:
            logger.info(prefix + f" iter {i} / {n_sample}")

        samples[:, i] = u
        log_measure[i] = pois.log_target(u)

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output = h5py.File(output_file, "w")
        output.attrs["nx"] = nx
        output.attrs["t_sample"] = t_sample
        output.attrs["n_sample"] = n_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nx", type=int)
    parser.add_argument("--n_sample", type=int)
    parser.add_argument("--sampler", type=str)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--n_inner", type=int, default=1)
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument("--cold_start", action="store_true")
    args = parser.parse_args()

    # make output directory
    path = os.path.dirname(args.output_file)
    os.makedirs(path, exist_ok=True)

    base, ext = os.path.splitext(args.output_file)
    logging.basicConfig(level=logging.INFO, filename=base + ".log")

    # log to console and file
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    if args.cold_start:
        start = "cold"
    else:
        start = "warm"
    logger.info("setting the start to be %s", start)

    eta = args.eta
    if eta is None:
        eta = (args.nx + 1)**(-2 / 3)
        logger.info("set eta to Roberts eta_optimal: %.8f", eta)
    else:
        logger.info("set eta to preset value: %.8f", eta)

    samplers = ["exact", "mala", "pmala", "pula-exact", "pula-lu"]

    if args.sampler == "exact":
        run_exact(args.nx, args.output_file, args.n_sample)
    elif args.sampler == "mala":
        run_mala(args.nx,
                 eta,
                 args.output_file,
                 args.n_sample,
                 n_inner=args.n_inner,
                 n_warmup=args.n_warmup,
                 start=start)
    elif args.sampler == "pmala":
        run_pmala(args.nx,
                  eta,
                  args.output_file,
                  args.n_sample,
                  n_inner=args.n_inner,
                  n_warmup=args.n_warmup,
                  start=start)
    elif args.sampler == "pula-lu":
        run_pula_lu(args.nx,
                    eta,
                    args.output_file,
                    args.n_sample,
                    n_inner=args.n_inner,
                    start=start)
    else:
        logger.info(f"Sampler not recognised: please use one of {samplers}")
