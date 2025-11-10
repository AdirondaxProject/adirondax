<p align="center">
  <a href="https://adirondax.readthedocs.io">
    <img src="docs/_static/adirondax-logo.svg" alt="adirondax" width="400"/>
  </a>
</p>

# adirondax

[![Repo Status][status-badge]][status-link]
[![PyPI Version Status][pypi-badge]][pypi-link]
[![Test Status][workflow-test-badge]][workflow-test-link]
[![Coverage][coverage-badge]][coverage-link]
[![Ruff][ruff-badge]][ruff-link]
[![asv][asv-badge]][asv-link]
[![Readthedocs Status][docs-badge]][docs-link]
[![License][license-badge]][license-link]

[status-link]:         https://www.repostatus.org/#active
[status-badge]:        https://www.repostatus.org/badges/latest/active.svg
[pypi-link]:           https://pypi.org/project/adirondax
[pypi-badge]:          https://img.shields.io/pypi/v/adirondax?label=PyPI&logo=pypi
[workflow-test-link]:  https://github.com/AdirondaxProject/adirondax/actions/workflows/test-package.yml
[workflow-test-badge]: https://github.com/AdirondaxProject/adirondax/actions/workflows/test-package.yml/badge.svg?event=push
[coverage-link]:       https://app.codecov.io/gh/AdirondaxProject/adirondax
[coverage-badge]:      https://codecov.io/github/adirondaxproject/adirondax/graph/adirondax-server/badge.svg
[ruff-link]:           https://github.com/astral-sh/ruff
[ruff-badge]:          https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[asv-link]:            https://adirondaxproject.github.io/adirondax-benchmarks/
[asv-badge]:           https://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat
[docs-link]:           https://adirondax.readthedocs.io
[docs-badge]:          https://readthedocs.org/projects/adirondax/badge
[license-link]:        https://opensource.org/licenses/Apache-2.0
[license-badge]:       https://img.shields.io/badge/License-Apache_2.0-blue.svg

A differentiable astrophysics simulator in JAX

Author: [Philip Mocz (@pmocz)](https://github.com/pmocz/)

⚠️ Adirondax is currently being built and is not yet ready for use. Check back later ⚠️

Adirondax is a high-performance scientific research software for conducting astrophysical and cosmological simulations. Being differentiable, Adirondax can seamlessly integrate with pipelines for inverse-problems, inference, optimization, and coupling to ML models. Adirondax is scalable on multiple GPUs.

Adirondax has a simpler companion project dedicated to Fuzzy Dark Matter simulations: [Jaxion](https://github.com/JaxionProject/jaxion)


## Install Adirondax

Install with 

```console
pip install adirondax
```

See the docs for more info on how to [build from source](https://adirondax.readthedocs.io/en/latest/pages/installation.html).


## Examples

Check out the [`examples/`](https://github.com/Adirondax/adirondax/tree/main/examples/) directory for demonstrations of using Adirondax.

<p align="center">
  <a href="https://github.com/Adirondax/adirondax/tree/main/examples/kelvin_helmholtz">
    <img src="examples/kelvin_helmholtz/movie.gif" alt="kelvin_helmholtz" width="128"/>
  </a>
  <a href="https://github.com/Adirondax/adirondax/tree/main/examples/logo_inverse_problem">
    <img src="examples/logo_inverse_problem/movie.gif" alt="logo_inverse_problem" width="128"/>
  </a>
  <a href="https://github.com/Adirondax/adirondax/tree/main/examples/orszag_tang">
    <img src="examples/orszag_tang/movie.gif" alt="orszag_tang" width="128"/>
  </a>
  <br>
</p>

## Links

* [Code repository](https://github.com/AdirondaxProject/adirondax) on GitHub (this page).
* [Documentation](https://adirondax.readthedocs.io) for up-to-date information about installing and using Adirondax.
* [asv benchmarks](https://adirondaxproject.github.io/adirondax-benchmarks) to view code timing and memory benchmarks (for Developers).


## Cite this repository

If you use this software, please cite it as below.

