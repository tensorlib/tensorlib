.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: tensorlib.datasets
This package features helpers to fetch larger datasets and parameters
commonly used by the machine learning community to benchmark algorithms on data
that comes from the 'real world'.

.. _sample_datasets:

Sample datasets
===============

Tensorlib embeds multidimensional sample data for testing algorithms
and creating examples.

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_bread

.. image:: ../auto_examples/images/plot_bread_001.png
   :target: ../auto_examples/plot_bread.html
   :scale: 50
   :align: center

.. topic:: Examples:

    * :ref:`plot_bread.py`


.. _larger_datasets:


Larger datasets
===============

Tensorlib also includes downloaders for larger datasets that
can be used for something closer to "real world" testing.


.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   fetch_decmeg

.. image:: ../auto_examples/images/plot_meg_001.png
   :target: ../auto_examples/datasets/plot_meg.html
   :scale: 50
   :align: center

.. topic:: Examples:

    * :ref:`plot_meg.py`
