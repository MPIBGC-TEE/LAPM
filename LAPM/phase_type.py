"""Module for phase-type distribution.

:math:`T` is supposed to be a phase-type distributed random variable.
"""
from __future__ import division

from sympy import symbols, Matrix, exp, ones, diag, simplify, eye
from math import factorial, sqrt
from numpy import array
from scipy.linalg import expm


#########################
# phase-type distribution
#########################


def z(B):
    """Return the (symbolic) vector of rates toward absorbing state.

    Args:
        B (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: :math:`z = -B^T\\,\\mathbf{1}`
    """
    o = ones(B.rows, 1)
    return -B.transpose()*o

def cum_dist_func(beta, B, Qt):
    """Return the (symbolic) cumulative distribution function of phase-type.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,B}`

    Returns:
        SymPy expression: cumulative distribution function of 
        PH(:math:`\\beta`, :math:`B`)

            :math:`F_T(t) = 1 - \\mathbf{1}^T\\,e^{t\\,B}\\,\\beta`
    """
    o = ones(1, B.cols)
    return 1 - (o * (Qt * beta))[0]

def expected_value(beta, B):
    """Return the (symbolic) expected value of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: 
            expected value of PH(:math:`\\beta`, :math:`B`)
            
            :math:`\\mathbb{E}[T] = -\\mathbf{1}^T\\,B^{-1}\\,\\beta`
    """
    return nth_moment(beta, B, 1)

def nth_moment(beta, B, n):
    """Return the (symbolic) ``n`` th moment of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
        n (positive int): order of the moment
    
    Returns:
        SymPy expression: ``n`` th moment of 
        PH(:math:`\\beta`, :math:`B`)
            
            :math:`\\mathbb{E}[T^n]=` 
            :math:`(-1)^n\\,n!\\,\\mathbf{1}^T\\,B^{-1}\\,\\beta`
    """
    o = ones(1, B.cols)
    return ((-1)**n*factorial(n)*o*(B.inv()**n)*beta)[0]

def variance(beta, B):
    """Return the (symbolic) variance of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: variance of PH(:math:`\\beta`, :math:`B`)
            :math:`\\sigma^2(T) = \\mathbb{E}[T^2] - (\\mathbb{E}[T])^2`

    See Also:
        | :func:`expected_value`: Return the (symbolic) expected value of the 
            phase-type distribution.
        | :func:`nth_moment`: Return the (symbolic) nth moment of the 
            phase-type distribution.
    """
    return nth_moment(beta, B, 2) - (expected_value(beta, B)**2)

def standard_deviation(beta, B):
    """Return the (symbolic) standard deviation of the phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
    
    Returns:
        SymPy expression: standard deviation of 
        PH(:math:`\\beta`, :math:`B`)
            
            :math:`\\sigma(T) = \\sqrt{\\sigma^2(T)}`

    See Also:
        :func:`variance`: Return the (symbolic) variance of the phase-type 
            distribution.
    """
    return sqrt(variance(beta, B))

def density(beta, B, Qt):
    """Return the (symbolic) probability density function of the 
    phase-type distribution.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,B}`

    Returns:
        SymPy expression: probability density function of 
        PH(:math:`\\beta`, :math:`B`)
            
            :math:`f_T(t) = z^T\\,e^{t\\,B}\\,\\beta`

    See Also:
        :func:`z`: Return the (symbolic) vector of external output rates.
    """
    return (z(B).transpose()*Qt*beta)[0]


def laplace(beta, B):
    """Return the symbolic Laplacian of the phase-type distribtion.

    Args:
        beta (SymPy dx1-matrix): initial distribution vector
        B (SymPy dxd-matrix): transition rate matrix

    Returns:
        SymPy expression: Laplace transform of the probability density of 
        PH(:math:`\\beta`, :math:`B`)
            
            :math:`L_T(s)=` 
            :math:`z^T\\,(s\\,I-B)^{-1}\\,\\beta`

    See Also:
        :func:`z`: Return the (symbolic) vector of external output rates.
    """
    s = symbols('s')
    
    return (z(B).transpose()*(  (s*eye(B.rows)-B).inv() )*beta)[0]



