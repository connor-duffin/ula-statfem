# sfmcmc: Statistical Finite elements via Markov Chain Monte Carlo

This package contains samplers for the statistical finite element method, as implemented for the Poisson equation, using Markov Chain Monte Carlo methods. Classes are included for both the prior and posterior distribution, with the user able to input their own data, or, generate synthetic data on the fly. Samplers include

* Unadjusted Langevin (ULA)
* Preconditioned ULA
* Metropolis adjusted Langevin Algorithm (MALA)
* Preconditioned MALA
* Preconditioned Crank Nicolson

With various preconditioning strategies available.

This package is currently quite bare-bones, implementing only the problems and samplers that we have investigated thus far.
Please let me know if there are any bugs! (connor.p.duffin (@) gmail.com)

## Installation

This package is not in `PyPI` yet, so just `cd` into this directory and run:

```{bash}
pip install -e sfmcmc
```

The `-e` flag creates an editable install; any local changes you make are then automatically loaded.

## Reference

A `bibtex` reference for this paper is:

```{bibtex}
@article{akyildiz2022statistical,
  author = {Akyildiz, \"{O}mer Deniz and Duffin, Connor and Sabanis, Sotirios and Girolami, Mark},
  title = {Statistical Finite Elements via Langevin Dynamics},
  journal = {SIAM/ASA Journal on Uncertainty Quantification},
  volume = {10},
  number = {4},
  pages = {1560-1585},
  year = {2022},
  doi = {10.1137/21M1463094}
}
```
