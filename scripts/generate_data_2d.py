import h5py

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


def generate_data(output_file):
    pois = PoissonUnitThetaPosterior(NX)
    pois.setup_G(sigma=SIGMA_XI)
    pois.setup_theta(RHO, ELL, method="kronecker")

    x_obs = lhs(2, NX_OBS)
    pois.setup_dgp(x_obs,
                   n_obs=NY_OBS,
                   sigma=SIGMA_Y,
                   scale_factor=SCALE_FACTOR)
    pois.generate_data()

    with h5py.File(output_file, "w") as f:
        f.attrs["x_obs"] = x_obs
        f.attrs["n_obs"] = NY_OBS
        f.attrs["sigma"] = SIGMA_Y
        f.attrs["scale_factor"] = SCALE_FACTOR

        f.create_dataset("y", data=pois.y)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file", default="data.h5")
    args = parser.parse_args()

    generate_data(args.output_file)
