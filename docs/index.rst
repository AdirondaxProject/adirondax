adirondax
=========

.. grid:: 1
    :class-container: color-cards

    .. grid-item-card:: Differentiable Astrophysics
      :columns: 12 12 12 12
      :class-card: adirondax-summary

      .. image:: _static/adirondax.png
         :alt: Adirondax logo
         :width: 96px
         :align: left

      **Adirondax** is a scientific research software for conducting astrophysical and cosmological simulations and solving inverse problems, implemented in JAX and utilizing automatic differentiation and multi-GPU acceleration.

⚠️ Adirondax is currently being built and is not yet ready for use. Check back later! ⚠️


.. grid:: 3
   :class-container: product-offerings
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Differentiable Physics
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Solvers in Adirondax are fully differentiable, enabling gradient-based inference, parameter optimization, and coupling to ML models directly through the simulation.

   .. grid-item-card:: Scalable Performance
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Built with JAX, Adirondax scales on multiple GPUs/TPUs, allowing large-scale simulations to run efficiently on modern accelerators.

   .. grid-item-card:: Modular & Composable
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Adirondax physics modules and processing tools are composable, making it easy to extend or embed into larger workflows.


.. grid:: 3
    :class-container: color-cards

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Installation
      :columns: 12 6 6 4
      :link: pages/installation
      :link-type: doc
      :class-card: installation

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Quickstart
      :columns: 12 6 6 4
      :link: pages/quickstart
      :link-type: doc
      :class-card: quickstart

    .. grid-item-card:: :material-regular:`library_books;2em` Examples
      :columns: 12 6 6 4
      :link: pages/examples
      :link-type: doc
      :class-card: examples



.. list-table::
   :widths: 32 32 32
   :header-rows: 0

   * - .. figure:: ../examples/kelvin_helmholtz/movie.gif
         :width: 300px
         :align: center
         :alt: kelvin_helmholtz
         :target: pages/examples.html#kelvin-helmholtz

     - .. figure:: ../examples/logo_inverse_problem/movie.gif
         :width: 300px
         :align: center
         :alt: logo_inverse_problem
         :target: pages/examples.html#logo-inverse-problem

     - .. figure:: ../examples/orszag_tang/movie.gif
         :width: 300px
         :align: center
         :alt: orszag_tang
         :target: pages/examples.html#orszag-tang


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    pages/installation
    pages/quickstart

.. toctree::
    :maxdepth: 1
    :caption: Tutorials & Examples

    pages/examples

.. toctree::
    :maxdepth: 1
    :caption: References

    pages/parameters
    pages/equations
    pages/api
    pages/developing
    pages/citing
    pages/about
