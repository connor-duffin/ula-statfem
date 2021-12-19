import os
import h5py
import jax.numpy as jnp

from pyDOE import lhs
from argparse import ArgumentParser
from sfmcmc.samplers import PoissonUnitThetaPosterior


NX = 32
SIGMA_XI = 0.05
RHO, ELL = 0.1, 0.2

NX_OBS = 128
NY_OBS = 100
SIGMA_Y = 1e-3
SCALE_FACTOR = 1.4


def sigmoid_obs(x):
    return 0.1 / (1 + jnp.exp(-100 * (x - 0.05)))


def generate_data(output_file, nonlinear_observation=False):
    pois = PoissonUnitThetaPosterior(NX)
    pois.setup_G(scale=SIGMA_XI)
    pois.setup_theta(RHO, ELL, method="kronecker")

    if nonlinear_observation:
        obs_function = None
    else:
        obs_function = sigmoid_obs

    x_obs = lhs(2, NX_OBS)
    pois.setup_dgp(x_obs,
                   n_obs=NY_OBS,
                   sigma=SIGMA_Y,
                   scale_factor=SCALE_FACTOR,
                   obs_function=obs_function)
    pois.generate_data()

    with h5py.File(output_file, "w") as f:
        f.attrs["x_obs"] = x_obs
        f.attrs["n_obs"] = NY_OBS
        f.attrs["sigma"] = SIGMA_Y
        f.attrs["scale_factor"] = SCALE_FACTOR
        f.attrs["nonlinear_observation"] = nonlinear_observation

        f.create_dataset("y", data=pois.y)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nonlinear_observation", action="store_true")
    parser.add_argument("--output_file", default="data.h5")
    args = parser.parse_args()

    path = os.path.dirname(args.output_file)
    os.makedirs(path, exist_ok=True)

    generate_data(args.output_file)
