# sfmcmc: Statistical Finite elements via Markov Chain Monte Carlo

This package contains samplers for the statistical finite element method, as implemented for the Poisson equation, using Markov Chain Monte Carlo methods. Classes are included for both the prior and posterior distribution, with the user able to input their own data, or, generate synthetic data on the fly. Samplers include

* Unadjusted Langevin (ULA)
* Preconditioned ULA
* Metropolis adjusted Langevin Algorithm (MALA)
* Preconditioned MALA
* Preconditioned Crank Nicolson

With various preconditioning strategies available.

This package is currently quite bare-bones, implementing only the problems and samplers that we have investigated thus far. Further development will be for additional problems and samplers.

Please let me know if there are any bugs! (connor.p.duffin (@) gmail.com)

## Installation

This package is not in `PyPI` yet, so just clone and install:

```{bash}
git clone https://github.com/connor-duffin/sfmcmc.git
cd sfmcmc
pip install -e sfmcmc
```

The `-e` flag creates an editable install; any local changes you make are then automatically loaded.

## Reference

A `bibtex` reference for this paper is:

```
@article{akyildiz2021statistical,
  blah blah blah
}
```
