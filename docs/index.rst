Linear Autonomous Pool Models (LAPM)
====================================

LAPM is a simple Python package to deal with linear autonomous pool models of the form

.. math:: \frac{d}{dt}\,\bf{x}(t) = \bf{A}\,\bf{x}(t) + \bf{u}.

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

    ~LAPM.PH
    ~LAPM.LinearAutonomousPoolModel
    ~LAPM.ExampleModels
    -----------
    ~LAPM.emanuel

----------------

Important Note
--------------

:math:`\bf{A=(a_{ij})}` has always to be an invertible *compartment matrix*:

* :math:`a_{ii}<0` for all :math:`i`
* :math:`a_{ij}\geq 0` for :math:`i\neq j`
* :math:`\sum\limits_{i=1}^d a_{ij}\leq 0` for all :math:`j`


---------------------------

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

