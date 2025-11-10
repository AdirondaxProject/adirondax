Quickstart
==========

To quickly try out Adirondax, install the package via pip:

.. code-block:: bash

    pip install adirondax

and run the Kelvin-Helmholtz Instability (``examples/kelvin_helmholtz/kelvin_helmholtz.py``):

.. code-block:: bash

    python kelvin_helmholtz.py

to produce a simulation run that looks something like:

.. image:: ../../examples/kelvin_helmholtz/movie.gif
   :width: 480px
   :align: center
   :alt: kelvin_helmholtz

The script that sets up and runs the problem is as follows:

.. literalinclude:: ../../examples/kelvin_helmholtz/kelvin_helmholtz.py
  :language: python


For more info
-------------

For info on how to install Adirondax with GPU support, see the :doc:`Installation <installation>` page.

For more examples of simulations, see the :doc:`Examples <examples>` page.
