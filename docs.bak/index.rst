Linear Autonomous Pool Models (LAPM)
====================================

`LAPM <https://github.com/MPIBGC-TEE/LAPM>`_ is a simple Python package to deal with linear autonomous pool models of the form

.. math:: \frac{d}{dt}\,x(t) = B\,x(t) + u.

It provides symbolic and numerical computation of

* steady state content
* steady state release
* transit time, system age, pool age

    * density
    * (cumulative distribution function)
    * mean
    * standard deviation
    * variance
    * higher order moments
    * (Laplace transforms)

Table of Contents
-----------------

.. autosummary::
    :toctree: _autosummary

    ~LAPM.phase_type
    ~LAPM.linear_autonomous_pool_model
    ~LAPM.dtmc
    ~LAPM.example_models
    -----------
    ~LAPM.emanuel

----------------

Important Note
--------------

:math:`\bf{B}=(b_{ij})` has always to be an invertible *compartmental matrix*:

* :math:`b_{ii}<0` for all :math:`i`
* :math:`b_{ij}\geq 0` for :math:`i\neq j`
* :math:`\sum\limits_{i=1}^d b_{ij}\leq 0` for all :math:`j`


---------------------------

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

