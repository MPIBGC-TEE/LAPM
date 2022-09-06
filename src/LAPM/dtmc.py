"""Module for discrete-time Markov chains (DTMCs)."""
from __future__ import division

from sympy import symbols, Matrix, diag, simplify, eye, Symbol, solve, Eq, ones

from .helpers import entropy


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class DTMC(object):
    """Class of discrete time Markov chains.

    Attributes:
        beta (SymPy dx1-matrix): initial distribution
        P (SymPy dxd-matrix): transition probability matrix
    """

    #fixme: test
    def __init__(self, beta, P):
        """Return a DTMC with initial distribution beta and transition 
        probability matrix P.

        Args: 
            beta (SymPy dx1-matrix): initial distribution
            P (Sympy dxd-matrix): transition probability matrix
        """
        self.beta = beta
        self.P = P

    @property
    def n(self):
        """int: Return the dimension of the Markov chain."""
        return self.P.rows

    @property
    def fundamental_matrix(self):
        """Return the (symbolic) fundamental matrix.
    
        Returns:
            SymPy or numerical dxd-matrix: 
                :math:`M=(I-P)^{-1}`

        Raises:
            Error: if :math:`\\operatorname{det}(I-P)=0`,
                   no absorbing Markov chain is given
        """
        try:
            M = (eye(self.P.rows)-self.P)**(-1)
        except ValueError as err:
            raise Error('P-I not invertible, probably no absorbing Markov chain'
                        ' given.') from err

        return simplify(M)

    @property
    def expected_number_of_jumps(self):
        """Return the (symbolic) expected number of jumps BEFORE absorption.

        Returns:
            SymPy expression or numerical value: 
                :math:`\\sum\\limits_{i=1}^n [M\\,\\beta]_i`

        See Also:
            :func:`fundamental_matrix`: 
            Return the (symbolic) fundamental matrix.

        Raises:
            Error: if :math:`\\operatorname{det}(I-P)=0`,
                   no absorbing Markov chain is given
        """
        n = self.n
        M = self.fundamental_matrix
        jumps = sum(M*self.beta) # sympy matrix multiplication
   
        return simplify(jumps)

    @property
    def stationary_distribution(self):
        """Return the (symbolic) stationary distribution.

        Returns:
            SymPy matrix: stationary distribution vector :math:`\\pi`
                :math:`P\\,\\pi=\\pi,\\quad \\sum\\limits_{j=1}^n \\pi_j=1`
        """
        n = self.n
        nu = Matrix(n, 1, [Symbol('nu_%s' % (j+1)) for j in range(n)])
        # create an additional line for sum(nu_j)=1
        l = [x for x in nu] + [1]
        v = Matrix(n+1, 1, l)

        l = [x for x in self.P]
        l += [1]*n
        P_extended = Matrix(n+1, n, l)

        # solve the system
        sol = solve(Eq(P_extended*nu, v), nu)
        #print('sol', sol)
    
        if sol == []: return None

        # make a vector out of dictionary solution
        #print(nu)
        l = [sol[x] for x in nu]
        return Matrix(l)

    @property
    def ergodic_entropy(self):
        """Return the ergodic entropy per jump.
        
        Returns:
            SymPy expression or float: 
                :math:`\\sum\\limits_{j=1}^n \\pi_j\\sum\\limits_{i=1}^n`
                :math:`-p_{ij}\\,\\log p_{ij}`

        See also:
            :func:`stationary_distribution`: 
            Return the (symbolic) stationary distribution.
        """
        P = self.P
        nu = self.stationary_distribution
        n = self.n
         
        theta = 0
        for j in range(n):
            x = 0
            for i in range(n):
                x += entropy(P[i,j])
    
            x *= nu[j]
            theta += x
    
        return theta

    
    
