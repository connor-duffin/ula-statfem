import os
import h5py
import time
import logging

import numpy as np

from argparse import ArgumentParser
from sfmcmc.samplers import PoissonUnitThetaPosterior
from run_samplers_2d import init_sample_array
from generate_data_2d import sigmoid_obs

logger = logging.getLogger(__name__)


def compute_map_estimate(pois,
                         stepsize=1e-4,
                         n_iter=10_000,
                         atol=1e-8,
                         epochs=1000):
    """
    Compute MAP estimate using a Gauss-Newton Hessian.
    """
    norm = np.linalg.norm

    u = pois.sample_posterior_exact()
    delta = np.zeros((pois.n_dofs, ))

    epoch_interval = n_iter // epochs
    log_targets = np.zeros((epochs, ))

    i_save = 0
    for i in range(1, n_iter + 1):
        u_next, delta = pois.newton_step(u, stepsize)

        if np.any(u_next > 1e8) or np.any(np.isnan(u_next)):
            logger.info(f"Optimizer diverged at iteration {i} --- exiting")
            break
        else:
            u[:] = u_next
            lp_next = pois.log_target(u)

        if i % epoch_interval == 0:
            log_targets[i_save] = lp_next
            logger.info(f"Norm of delta: {norm(delta):.6e}")
            i_save += 1

        if norm(delta) <= atol:
            logger.info(f"Converged at iteration {i}")
            break

    if i == n_iter:
        logger.warning("Convergence not necessarily achieved")

    return u


def init_pois(nx, data_file, start="cold", obs_function=None):
    pois = PoissonUnitThetaPosterior(nx)
    pois.setup_G(scale=0.05)
    pois.setup_theta(0.2, 0.2, method="kronecker")

    with h5py.File(data_file, "r") as f:
        pois.setup_dgp(x_obs=f.attrs["x_obs"],
                       n_obs=f.attrs["n_obs"],
                       sigma=f.attrs["sigma"],
                       scale_factor=f.attrs["scale_factor"],
                       obs_function=obs_function)
        pois.setup_jax()
        pois.load_data(f["y"])

    if obs_function is not None:
        u_optim = compute_map_estimate(pois,
                                       stepsize=1e-2,
                                       atol=1e-8,
                                       n_iter=50_000)
        pois.setup_pc_post(u_optim)

    if start == "cold":
        pois.u[:] = 0.
    elif start == "warm":
        if obs_function is None:
            pois.u[:] = pois.sample_posterior_exact()
        else:
            pois.u[:] = u_optim
    else:
        raise ValueError("Not able to recognise initialization setting.")

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
            obs_function,
            output_file=None,
            n_warmup=0,
            n_inner=1):
    prefix = "(ula)"
    pois = init_pois(nx, data_file, start="warm", obs_function=obs_function)
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
        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        pois.ula_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            pois.ula_step(eta, fixed_theta=True)

        # cache in case of failure (faster than every `thin`th)
        if i % 1000 == 0 and output_file is not None:
            u_temp[:] = pois.u
            eta_temp[:] = eta
            i_temp[:] = i

        if i == n_warmup:
            t_start = time.time()

        if i % 100 == 0:
            logger.info(prefix + f" iteration {i + 1} of {n_sample}")

        if np.any(pois.u > 1e6) or np.any(np.isnan(pois.u)):
            logger.info(f"ULA failed at iteration {i}")
            raise OverflowError

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
             obs_function,
             output_file=None,
             n_warmup=0,
             n_inner=1):
    prefix = "(pula)"
    pois = init_pois(nx, data_file, start="warm", obs_function=obs_function)

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
        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        pois.ula_step(eta, fixed_theta=False, pc=True)

        for j in range(n_inner - 1):
            pois.ula_step(eta, fixed_theta=True, pc=True)

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
             obs_function,
             output_file=None,
             n_warmup=0,
             n_inner=1):
    prefix = "(mala)"
    pois = init_pois(nx, data_file, start="warm", obs_function=obs_function)

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
        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        accepted = pois.mala_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.mala_step(eta, fixed_theta=True)

        if accepted:
            n_accept += 1

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
              obs_function,
              output_file,
              n_warmup=0,
              n_inner=1):
    prefix = "(pmala)"
    pois = init_pois(nx, data_file, start="warm", obs_function=obs_function)

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
        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        accepted = pois.mala_step(eta, fixed_theta=False, pc=True)

        for j in range(n_inner - 1):
            accepted = pois.mala_step(eta, fixed_theta=True, pc=True)

        if accepted:
            n_accept += 1

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


def run_pcn(nx,
            n_sample,
            eta,
            data_file,
            obs_function,
            output_file,
            n_warmup=0,
            n_inner=1):
    prefix = "(pcn)"
    pois = init_pois(nx, data_file, start="warm", obs_function=obs_function)

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
        samples[:, i] = pois.u
        log_measure[i] = pois.log_likelihood(pois.u)

        accepted = pois.pcn_step(eta, fixed_theta=False)

        for j in range(n_inner - 1):
            accepted = pois.pcn_step(eta, fixed_theta=True)

        if accepted:
            n_accept += 1

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
    parser.add_argument("--nonlinear_observation", action="store_true")
    args = parser.parse_args()
    logger.info(args)

    # make output directory
    path = os.path.dirname(args.output_file)
    os.makedirs(path, exist_ok=True)

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

    if args.nonlinear_observation:
        obs_function = sigmoid_obs
    else:
        obs_function = None

    sampler = args.sampler.lower()
    if sampler == "exact":
        run_exact(args.nx, args.n_sample, data_file, output_file)
    elif sampler == "ula":
        run_ula(args.nx, args.n_sample, eta, data_file, obs_function,
                output_file, args.n_warmup, args.n_inner)
    elif sampler == "pula":
        run_pula(args.nx, args.n_sample, eta, data_file, obs_function,
                 output_file, args.n_warmup, args.n_inner)
    elif sampler == "mala":
        run_mala(args.nx, args.n_sample, eta, data_file, obs_function,
                 output_file, args.n_warmup, args.n_inner)
    elif sampler == "pmala":
        run_pmala(args.nx, args.n_sample, eta, data_file, obs_function,
                  output_file, args.n_warmup, args.n_inner)
    elif sampler == "pcn":
        run_pcn(args.nx, args.n_sample, eta, data_file, obs_function,
                output_file, args.n_warmup, args.n_inner)
    else:
        print(f"Sampler argument '{sampler}' not recognised")
