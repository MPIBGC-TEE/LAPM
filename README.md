[![Travis-CI Build Status](https://travis-ci.org/MPIBGC-TEE/LAPM.svg?branch=master)](https://travis-ci.org/MPIBGC-TEE/LAPM)
# LAPM: Linear Autonomous Pool Models

This is a Python3 package for linear autonomous pool models.
It provides a class LinearAutonomousPoolModels that allows 
computation of transit time, system age, and pool ages.

Since transit time and system age are phase-type distributed, the computations 
of them rely on properties of this distribution. They are treated in a separate 
module.

---

<!--[Documentation](http://lapm.readthedocs.io/en/latest/)-->
[Documentation](https://mpibgc-tee.github.io/LAPM/).
---

Installation simply via the install script `install.sh`.
Use either `develop` or `install` as additional parameter, it will be 
passed to the `python3 setup.py` call.

---

If you do not use the install script, be sure to have installed the following 
packages:

+ numpy
+ scipy
+ sympy
+ matplotlib

