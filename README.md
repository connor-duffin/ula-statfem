# Langevin samplers for statFEM

Scripts and notebooks for simulations on unadjusted Langevin (ULA) and preconditioned unadjusted Langevin samplers (pULA), using `Fenics` for FEM. Based on the work of Akyildiz, Duffin, Sabanis, and Girolami (2021).

## Setup

Included in this repository is the `environment.yml` file, that contains all the necessary packages that need to be installed in a conda environment in order to be run our code. Let me know if any are missing!

To setup the environment run
```{bash}
conda env create --file environment.yml
```

which will install all the required packages. The main thing is `Fenics`, a finite element library that allows us to build the necessary matrices for our PDE problem. It also does *way* more than this, but for our purposes, this is really all we use it for.

Then, with the conda environment active, to install the package that contains the Langevin samplers, run `pip3 install -e sfmcmc`, which will install a local copy of the `sfmcmc` package. This will enable you to now run the notebooks and scripts, as well as generate the plots.

## Notebooks

Probably the thing of most interest are the notebooks, which are in the `docs/` directory. To use the notebooks, run `jupyter notebook` from this directory, and navigate into the `docs/` subdirectory from the Jupyter interface. Then choose the notebook of interest, and it should be good to go.

All notebooks have been designed to be run-able from this directory, so hopefully this works OK. Let me know if there are any issues.

## Code layout

The main code is hosted in the package `sfmcmc`, which contains all the code for ULA, MALA, etc. We also have notebooks, hosted in the `docs/` directory, which illustrate the algorithms. To replicate our results, we have also included various running and plotting scripts, and the `Makefile` which we use to run the results.

The `Makefile` has the full details of the simulations, with simulation settings and parameters, etc.

## Testing

Included in the package is a set of unit tests, under the `sfmcmc/tests` directory. To run these tests, first make sure the `langevin-statfem` environment is active. Then from the `sfmcmc` directory, run

```{bash}
python3 -m pytest .
```

This will run unit tests for the modules `samplers` and `utils`.
