"""Module for phase-type distribution."""
from __future__ import division

from sympy import symbols, Matrix, exp, ones, diag, simplify, eye
from math import factorial, sqrt
from numpy import array
from scipy.linalg import expm


#########################
# phase-type distribution
#########################


def z(A):
    """Return the (symbolic) vector of rates toward absorbing state.

    Args:
        A (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: :math:`\\bf{z} = -B^T\\,\\bf{1}`
    """
    o = ones(A.rows, 1)
    return -A.transpose()*o

def cum_dist_func(beta, A, Qt):
    """Return the (symbolic) cumulative distribution function of phase-type.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,\\bf{A}}`

    Returns:
        SymPy expression: cumulative distribution function of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`F_T(t) = 1 - \\bf{1}^T\\,e^{t\\,\\bf{A}}\\,\\bf{\\beta}`
    """
    o = ones(1, A.cols)
    return 1 - (o * (Qt * beta))[0]

def expected_value(beta, A):
    """Return the (symbolic) expected value of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: expected value of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`\mathbb{E}[T] = -\\bf{1}^T\\,\\bf{A}^{-1}\\,\\bf{\\beta}`
    """
    return nth_moment(beta, A, 1)

def nth_moment(beta, A, n):
    """Return the (symbolic) nth moment of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
        n (positive int): order of the moment
    
    Returns:
        SymPy expression: nth moment of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`\mathbb{E}[T^n] = (-1)^n\\,n!\\,\\bf{1}^T\\,\\bf{A}^{-1}\\,\\bf{\\beta}`
    """
    o = ones(1, A.cols)
    return ((-1)**n*factorial(n)*o*(A**-n)*beta)[0]

def variance(beta, A):
    """Return the (symbolic) variance of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: variance of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`\\sigma^2(T) = \\mathbb{E}[T^2] - (\\mathbb{E}[T])^2`

    See Also:
        | :func:`expected_value`: Return the (symbolic) expected value of the phase-type distribution.
        | :func:`nth_moment`: Return the (symbolic) nth moment of the phase-type distribution.
    """
    return nth_moment(beta, A, 2) - (expected_value(beta, A)**2)

def standard_deviation(beta, A):
    """Return the (symbolic) standard deviation of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: standard deviation of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`\\sigma(T) = \\sqrt{\\sigma^2(T)}`

    See Also:
        :func:`variance`: Return the (symbolic) variance of the phase-type distribution.
    """
    return sqrt(variance(beta, A))

def density(beta, A, Qt):
    """Return the (symbolic) probability density of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,\\bf{A}}`

    Returns:
        SymPy expression: probability density of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`f_T(t) = \\bf{z}^T\\,e^{t\\,\\bf{A}}\\,\\bf{\\beta}`

    See Also:
        :func:`z`: Return the (symbolic) vector of external output rates.
    """
    return (z(A).transpose()*Qt*beta)[0]


def laplace(beta, A):
    """Return the symbolic Laplacian of the phase-type distribtion.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        A (SymPy dxd-matrix): transition rate matrix

    Returns:
        SymPy expression: Laplace transform of the probability density of PH(:math:`\\bf{\\beta}`, :math:`\\bf{A}`)
            :math:`L_T(s) = \\bf{z}^T\\,(s\\,\\bf{I}-\\bf{A})^{-1}\\,\\bf{\\beta}`

    See Also:
        :func:`z`: Return the (symbolic) vector of external output rates.
    """
    s = symbols('s')
    
    return (z(A).transpose()*((s*eye(A.rows)-A)**-1)*beta)[0]



