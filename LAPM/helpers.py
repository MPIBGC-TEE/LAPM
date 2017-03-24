"""Module for package-wide used helper functions."""
from __future__ import division

from sympy import log


def entropy(p):
    """Return the entropy of a Bernoulli random variable with success 
    probability p."""
    if (p == 0) or (p == 1): return 0
    return -p * log(p)


