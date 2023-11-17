"""Module for package-wide used helper functions."""
from __future__ import division

import numpy as np
from sympy import Matrix, log


def entropy(p):
    """Return the entropy of a Bernoulli random variable with success
    probability p."""
    if (p == 0) or (p == 1):
        return 0
    return -p * log(p)


def vector_entropy(v):
    """Return the entropy of a probability vector."""
    return sum([entropy(p) for p in v])


def draw_multinomial_from_sympy(p: Matrix) -> int:
    "Sample from multinomial distribution from sympy vector."
    j = np.random.multinomial(1, np.array(p).astype(float).flatten(), size=1)[0]
    return int(np.where(j == 1)[0].astype(float))
