import os
import h5py
import time
import logging

import numpy as np

from argparse import ArgumentParser
from sfmcmc.samplers import PoissonUnitThetaPosterior
from run_samplers_2d import init_sample_array

logger = logging.getLogger(__name__)


# TODO: add option for cold/warm start
def init_pois(nx, data_file):
    pois = PoissonUnitThetaPosterior(nx)
    pois.setup_G(sigma=0.05)
    pois.setup_theta(0.2, 0.2, method="kronecker")

    with h5py.File(data_file, "r") as f:
        pois.setup_dgp(x_obs=f.attrs["x_obs"],
                       n_obs=f.attrs["n_obs"],
                       sigma=f.attrs["sigma"],
                       scale_factor=f.attrs["scale_factor"])
        pois.load_data(f["y"])

    pois.setup_pc_post()
    pois.u[:] = 0.
    return pois


def adapt_stepsize(stepsize, acceptance_ratio, lower, upper, scale=0.1):
    if acceptance_ratio >= upper:
        stepsize *= (1 + scale)
    elif acceptance_ratio <= lower:
        stepsize *= (1 - scale)

    return stepsize


def run_exact(nx, n_sample, data_file, output_file):
    prefix = "(exact)"
    pois = init_pois(nx, data_file)

    pois.u = np.zeros((pois.n_dofs, ))
    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)

    t_start = time.time()
    logger.info(prefix + " starting sampling now")
    for i in range(n_sample):
        u = pois.sample_posterior_exact()

        samples[:, i] = u
        log_measure[i] = pois.log_likelihood(u)

        if i % 100 == 0:
            logger.info(prefix + f" Iteration {i + 1} of {n_sample}")

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    output = h5py.File(output_file, "w")
    output.attrs["nx"] = nx
    output.attrs["t_sample"] = t_sample

    output.create_dataset("samples", data=samples)
    output.create_dataset("log_measure", data=log_measure)


def run_ula(nx,
            n_sample,
            eta,
            data_file,
            output_file=None,
            n_warmup=0,
            n_inner=1):
    prefix = "(ula)"
    pois = init_pois(nx, data_file)
    logger.info(prefix + f" Using eta = {eta:.6e}")

    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    if output_file is not None:
        # cache datasets for initialization
        output = h5py.File(output_file, "w")
        u_temp = output.create_dataset("u_temp", shape=(pois.n_dofs, ))
        eta_temp = output.create_dataset("eta_temp", shape=(1, ))
        i_temp = output.create_dataset("i_temp", shape=(1, ))

        # metadata
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta

    logger.info(prefix + " starting sampling")
    for i in range(n_sample):
        pois.ula_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            pois.ula_step(eta, fixed_theta=True)

        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i == n_warmup:
            t_start = time.time()

        if i % 100 == 0:
            logger.info(prefix + f" iteration {i + 1} of {n_sample}")

        if np.any(pois.u > 1e6):
            logger.info(f"ULA failed at iteration {i}")
            break

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output.attrs["eta"] = eta
        output.attrs["method"] = "mala"
        output.attrs["t_sample"] = t_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)
        output.close()


def run_pula(nx,
             n_sample,
             eta,
             data_file,
             output_file=None,
             n_warmup=0,
             n_inner=1):
    prefix = "(pula)"
    pois = init_pois(nx, data_file)

    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    if output_file is not None:
        # cache datasets for initialization
        output = h5py.File(output_file, "w")
        u_temp = output.create_dataset("u_temp", shape=(pois.n_dofs, ))
        eta_temp = output.create_dataset("eta_temp", shape=(1, ))
        i_temp = output.create_dataset("i_temp", shape=(1, ))

        # metadata
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta

    logger.info(prefix + " starting sampling")
    for i in range(n_sample):
        pois.pula_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            pois.pula_step(eta, fixed_theta=True)

        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i == n_warmup:
            t_start = time.time()

        if i % 100 == 0:
            logger.info(prefix + f" iteration {i + 1} of {n_sample}")

        if np.any(pois.u > 1e6):
            logger.info(prefix + f" failed at iteration {i}")
            break

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output.attrs["eta"] = eta
        output.attrs["method"] = "mala"
        output.attrs["t_sample"] = t_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)
        output.close()


def run_mala(nx,
             n_sample,
             eta,
             data_file,
             output_file=None,
             n_warmup=0,
             n_inner=1):
    prefix = "(mala)"
    pois = init_pois(nx, data_file)

    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    if output_file is not None:
        # cache datasets for initialization
        output = h5py.File(output_file, "w")
        u_temp = output.create_dataset("u_temp", shape=(pois.n_dofs, ))
        eta_temp = output.create_dataset("eta_temp", shape=(1, ))
        i_temp = output.create_dataset("i_temp", shape=(1, ))

        # metadata
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta
        output.attrs["n_warmup"] = n_warmup

    n_accept = 0
    logger.info(prefix + " starting sampling")
    for i in range(n_sample):
        accepted = pois.mala_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.mala_step(eta, fixed_theta=True)

        if accepted:
            n_accept += 1

        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        if i == n_warmup:
            t_start = time.time()

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i % 100 == 0:
            acceptance_ratio = n_accept / 100
            logger.info(prefix + f" iteration {i + 1} / {n_sample}" +
                        f" acceptance ratio: {acceptance_ratio}" +
                        f" stepsize: {eta:.6e}")
            if i < n_warmup:
                eta = adapt_stepsize(eta, acceptance_ratio, 0.4, 0.6)

            n_accept = 0

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output.attrs["eta"] = eta
        output.attrs["method"] = "mala"
        output.attrs["t_sample"] = t_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)
        output.close()


def run_pmala(nx,
              n_sample,
              eta,
              data_file,
              output_file,
              n_warmup=0,
              n_inner=1):
    prefix = "(pmala)"
    pois = init_pois(nx, data_file)

    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    if output_file is not None:
        # cache datasets for initialization
        output = h5py.File(output_file, "w")
        u_temp = output.create_dataset("u_temp", shape=(pois.n_dofs, ))
        eta_temp = output.create_dataset("eta_temp", shape=(1, ))
        i_temp = output.create_dataset("i_temp", shape=(1, ))

        # metadata
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta  # pre-run
        output.attrs["n_warmup"] = n_warmup

    n_accept = 0
    logger.info(prefix + " starting sampling")
    for i in range(n_sample):
        accepted = pois.pmala_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.pmala_step(eta, fixed_theta=True)

        if accepted:
            n_accept += 1

        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        if i == n_warmup:
            t_start = time.time()

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i % 100 == 0:
            acceptance_ratio = n_accept / 100
            logger.info(prefix + f" iteration {i + 1} / {n_sample}" +
                        f" acceptance ratio: {acceptance_ratio}" +
                        f" stepsize: {eta:.6e}")
            if i < n_warmup:
                eta = adapt_stepsize(eta, acceptance_ratio, 0.4, 0.6)
            n_accept = 0

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output.attrs["eta"] = eta
        output.attrs["method"] = "pmala"
        output.attrs["t_sample"] = t_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)
        output.close()


def run_pcn(nx, n_sample, eta, data_file, output_file, n_warmup=0, n_inner=1):
    prefix = "(pcn)"
    pois = init_pois(nx, data_file)

    samples = init_sample_array(n_sample, 1, pois.n_dofs)
    log_measure = init_sample_array(n_sample, 1, 1)
    logger.info("storing output samples in array of size %s", samples.shape)

    if output_file is not None:
        # cache datasets for initialization
        output = h5py.File(output_file, "w")
        u_temp = output.create_dataset("u_temp", shape=(pois.n_dofs, ))
        eta_temp = output.create_dataset("eta_temp", shape=(1, ))
        i_temp = output.create_dataset("i_temp", shape=(1, ))

        # metadata
        output.attrs["nx"] = nx
        output.attrs["eta"] = eta
        output.attrs["n_warmup"] = n_warmup

    n_accept = 0
    for i in range(n_sample):
        accepted = pois.pcn_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.pcn_step(eta, fixed_theta=True)

        if accepted:
            n_accept += 1

        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        if i == n_warmup:
            t_start = time.time()

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i % 100 == 0:
            acceptance_ratio = n_accept / 100
            logger.info(prefix + f" iteration {i + 1} / {n_sample}" +
                        f" acceptance ratio: {acceptance_ratio}" +
                        f" stepsize: {eta:.6e}")
            if i < n_warmup:
                eta = adapt_stepsize(eta, acceptance_ratio, 0.2, 0.4)
            n_accept = 0

    t_end = time.time()
    t_sample = t_end - t_start
    logger.info(prefix + f" Took {t_sample:.6f} s to sample")

    if output_file is None:
        return samples, log_measure
    else:
        output.attrs["eta"] = eta
        output.attrs["method"] = "pcn"
        output.attrs["t_sample"] = t_sample
        output.create_dataset("samples", data=samples)
        output.create_dataset("log_measure", data=log_measure)
        output.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nx", type=int)
    parser.add_argument("--n_sample", type=int)
    parser.add_argument("--sampler", type=str)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument("--n_inner", type=int, default=1)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--data_file", type=str)
    args = parser.parse_args()
    logger.info(args)

    base, ext = os.path.splitext(args.output_file)
    logging.basicConfig(level=logging.INFO, filename=base + ".log")

    # log to console and file
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    eta = args.eta
    if eta is None:
        eta = (args.nx + 1)**(-2 / 3)
        logger.info(f"Using Roberts optimal eta = {eta:.6e}")
    else:
        logger.info(f"Using specified eta = {eta:.6e}")

    output_file = args.output_file
    data_file = args.data_file
    logger.info("Storing output to %s, using data in %s", output_file,
                data_file)

    sampler = args.sampler.lower()
    if sampler == "exact":
        run_exact(args.nx, args.n_sample, data_file, output_file)
    elif sampler == "ula":
        run_ula(args.nx, args.n_sample, eta, data_file, output_file,
                args.n_warmup, args.n_inner)
    elif sampler == "pula":
        run_pula(args.nx, args.n_sample, eta, data_file, output_file,
                 args.n_warmup, args.n_inner)
    elif sampler == "mala":
        run_mala(args.nx, args.n_sample, eta, data_file, output_file,
                 args.n_warmup, args.n_inner)
    elif sampler == "pmala":
        run_pmala(args.nx, args.n_sample, eta, data_file, output_file,
                  args.n_warmup, args.n_inner)
    elif sampler == "pcn":
        run_pcn(args.nx, args.n_sample, eta, data_file, output_file,
                args.n_warmup, args.n_inner)
    else:
        print(f"Sampler argument '{sampler}' not recognised")
