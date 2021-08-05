"""Module for linear autonomous pool models."""
from __future__ import division

from sympy import symbols, Matrix, exp, ones, diag, simplify, eye, log
from math import factorial, sqrt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from . import phase_type
from .dtmc import DTMC
from .helpers import entropy


#############################################
# Linear autonomous compartment model class #
# d/dt x(t) = Bx(t) + u                     #
#############################################


def _age_vector_dens(u, B, Qt):
    """Return the (symbolic) probability density vector of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,B}`

    Returns:
        SymPy dx1-matrix: probability density vector of the compartment ages
        :math:`f_a(y) = (X^\\ast)^{-1}\\,e^{y\\,B}\\,u`
    """
    xss = -B.inv()*u
    X = diag(*xss)
    
    return X.inv()*Qt*u


def _age_vector_cum_dist(u, B, Qt):
    """Return the (symbolic) cumulative distribution function vector of the 
    compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,B}`

    Returns:
        SymPy dx1-matrix: cumulative distribution function vector of the 
        compartment ages
            
        :math:`f_a(y)=(X^\\ast)^{-1}\\,B^{-1}\\,(e^{y\\,B}-I)\\,u`
    """
    d = B.rows
    xss = -B.inv()*u
    X = diag(*xss)

    return X.inv()*B.inv()*(Qt-eye(d))*u


def _age_vector_nth_moment(u, B, n):
    """Return the (symbolic) vector of nth moments of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix
        n (positive int): order of the moment

    Returns:
        SymPy dx1-matrix: vector of nth moments of the compartment ages
        :math:`\\mathbb{E}[a^n]=(-1)^n\\,n!\\,X^\\ast)^{-1}\\,B^{-n}\\,x^\\ast`
    
    See Also:
        :func:`_age_vector_exp`: Return the (symbolic) vector of expected values
        of the compartment ages.
    """
    xss = -B.inv()*u
    X = diag(*xss)

    return (-1)**n*factorial(n)*X.inv()*(B.inv()**n)*xss


def _age_vector_exp(u, B):
    """Return the (symbolic) vector of expected values of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of expected values of the compartment ages
        :math:`\\mathbb{E}[a] = -(X^\\ast)^{-1}\\,B^{-1}\\,x^\\ast`
    
    See Also:
        :func:`_age_vector_nth_moment`: Return the (symbolic) vector of 
        ``n`` th moments of the compartment ages.
    """
    return _age_vector_nth_moment(u, B, 1)


def _age_vector_variance(u, B):
    """Return the (symbolic) vector of variances of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of variances of the compartment ages
        :math:`\\sigma^2(a) = \\mathbb{E}[a^2] - (\\mathbb{E}[a])^2`
        component-wise
    
    See Also:
        | :func:`_age_vector_exp`: Return the (symbolic) vector of expected 
            values of the compartment ages.
        | :func:`_age_vector_nth_moment`: Return the (symbolic) vector of 
            ``n`` th moments of the compartment ages.
    """
    sv = _age_vector_nth_moment(u, B, 2)
    ev = _age_vector_exp(u, B)
    vl = [sv[i] - ev[i]**2 for i in range(sv.rows)]
    
    return Matrix(sv.rows, 1, vl)
    

def _age_vector_sd(u, B):
    """Return the (symbolic) vector of standard deviations of the 
    compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        B (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of standard deviations of the compartment ages
        :math:`\\sigma(a) = \\sqrt{\\sigma^2(a)}` component-wise
    
    See Also:
        :func:`_age_vector_variance`: Return the (symbolic) vector of variances
        of the compartment ages.
    """
    sdl = [sqrt(e) for e in _age_vector_variance(u, B)]
    
    return Matrix(len(sdl), 1, sdl)


def _generalized_inverse_CDF(CDF, u, start_dist=1e-4, tol=1e-8):
    """Compute generalized inverse of a cumulative distribution function.

    Can be used for quantile computation or generation of random variables

    Args:
        CDF (Python function): cumulative distribution function
        start_dist (float): how far right from 0 to start looking for the 
            bisection interval,
            defaults to 1e-4
        tol (float): tolerance for brentq algorithm of numerical root search,
            defaults to 1e-8

    Returns:
        int: smallest :math:`y` such that :math:`\\operatorname{CDF}(y)\\geq u`
    """
    def f(a):
        res = u-CDF(a)
        return res

    x1 = start_dist
 
    # go so far to the right such that CDF(x1)>u, the bisect in interval [0, x1]
    y1 = f(x1)
    while y1 >= 0:
        x1 = x1*2 + 0.1
        y1 = f(x1)
    
    if np.isnan(y1):
        res = np.nan
    else:
        res = brentq(f, 0, x1, xtol=tol)
    
    return res


def create_random_probability_vector(d: int, p: float) -> np.ndarray:
    """Create a random probability vector.
    
    Args:
        d: dimension of the random vector
        p: probability of setting elements to nonzero value
        
    Returns:
        random probability vector `v` s.t.
        - :math:`v_i \geq 0`
        - :math:`\mathbb{P}(v_i>0) = p`
        - :math:`\sum_i v_i=1`
    """
    v = np.random.binomial(1, p, d)
    v = v * np.random.uniform(0, 1, d)
    if v.sum():
        v = v / v.sum()
        return np.random.permutation(v)
    else:
        # ensure that at least one entry is greater than 0
        c = np.random.choice(np.arange(d))
        v[c] = 1.0
        return v


def create_random_compartmental_matrix(d: int, p: float) -> np.ndarray:
    r"""Create a random invertible compartmental matrix.
    
    Args:
        d: dimension of the matrix (number of pools)
        p: probability of existence of non-necessary connections
        
    Returns:
        random invertible compartmental matrix `B` s.t.

        - all diagonal entries are nonpositive
        - all off-diagonal entries are nonnegative
        - all column sums are nonpositive
        - :math:`B` is invertible
        - :math:`-B_{ii} <= 1`
        - :math:`\mathbb{P}(B_{ij}>0) = p` for :math:`i \neq j`
        - :math:`\mathbb{P}(z_j)>0) = p`, where :math:`z_j = -\sum_i B_{ij}`
    """
    V = np.zeros((d+1, d+1))

    # add pools step by step, always keep the new pools connected to the 
    # previous smaller system
    # pool 0 is the output pool
    
    # build adjacency matrix
    for n in range(1, d+1):
        # random connections to smaller output-connected system
        V[:n, n] = np.random.binomial(1, p, n)
        
        # random connections from smaller output-connected system
        V[n, :n] = np.random.binomial(1, p, n)
        
        # ensure connection to smaller output-connected system
        c = np.random.choice(np.arange(n))
        V[c, n] = 1

    # create B from adjacency matrix,
    # repeat until det(B) is significantly different from 0
    x = 0
    while x < 1e-08:
        # make random speeds from adjacencies
        # (1 - uniform) to make sure the value will not be 0
        # with a 0 value an essential connection could get lost
        B = V * (1 - np.random.uniform(0, 1, (d+1)**2).reshape((d+1, d+1)))
    
        # build diagonals
        for j in range(B.shape[1]):
            B[j, j] = -B[:, j].sum()
        
        # ignore output pool 0
        B = B[1:, 1:]
        
        x = np.abs(np.linalg.det(B))
    return B


############################################################################


class Error(Exception):
    """Generic error occuring in this module."""
    pass

class NonInvertibleCompartmentalMatrix(Exception):
    pass


class LinearAutonomousPoolModel(object):
    """General class of linear autonomous compartment models.

    :math:`\\frac{d}{dt} x(t) = B\\,x(t) + u`

    Notes:
        - symbolic matrix exponential Qt = :math:`e^{t\\,B}` 
            cannot be computed automatically
        - for symbolic computations it has to be given manually: model.Qt = ...
        - otherwise only numerical computations possible
        - symbolical computations can take very long, in particular
          for complicated systems. An enforced purely numerical
          treatment could then be the right choice; choose
          ``force_numerical=True``
        - number of pools denoted by :math:`d`
        - steady state vector denoted by 
          :math:`x^\\ast=-B^{-1}\\,u`
        - normalized steady state vector denoted by
          :math:`\\eta=\\frac{x^\\ast}{\\|x^\\ast\\|}`
        - norm of a vector :math:`v` given by
          :math:`\\|\\mathbf{v}\\| = \\sum\\limits_{i=1}^d |v_i|`

    Attributes:
        u (SymPy dx1-matrix): The model's external input vector.
        B (SymPy dxd-matrix): The model's compartment matrix.
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,B}`
    """


    def __init__(self, u, B, force_numerical=False):
        """Return a linear autonomous compartment model with external 
        inputs u and (invertible) compartment matrix B.

        Symbolical computations can take very long, in particular
        for complicated systems. An enforced purely numerical
        treatment could then be the right choice.

        Args:
            u (SymPy dx1-matrix): external input vector
            B (SymPy dxd-matrix): invertible compartment matrix
            force_numerical (boolean, optional): 
                if True do not try symbolical computations,
                defaults to False

        Remark:
            While there are compartmental 
            systems that have singular matrices (systems with traps), most of the 
            methods of this class are only correct for invertible B. 
            We therefore reject singular matrices alltogether.
        """
        # cover one-dimensional input
        if (not hasattr(u, 'is_Matrix')) or (not u.is_Matrix): 
            u = Matrix(1, 1, [u])
        if (not hasattr(B, 'is_Matrix')) or (not B.is_Matrix): 
            B = Matrix(1, 1, [B])
  
        try:
            B.inv()
        except ValueError as e:
            print(B)
            raise NonInvertibleCompartmentalMatrix("""
            The matrix B is not invertible. 
            While there are compartmental systems that have singular matrices 
            (systems with traps), most of the methods of this class are only 
            correct for invertible B. 
            We therefore reject singular matrices alltogether.
            """)

        self.B = B
        self.u = u  
        # compute matrix exponential if no symbols are involved
        if (not force_numerical) and ((B.is_Matrix) and (len(B.free_symbols) == 0)):
            t = symbols('t')
            self.Qt = exp(t*B)

    @classmethod
    def from_random(cls, d: int, p: float):
        """Create a random compartmental system.

    
        Args:
            d: dimension of the matrix (number of pools)
            p: probability of having a connection between two pools
                and of nonzero value in input vector
        
        Returns:
            randomly generated compartmental system
            
            - `beta`: from :func:`create_random_probability_vector`
            - `B: from :func:`~create_random_compartmental_matrix`
        """
        beta = Matrix(create_random_probability_vector(d, p))
        B = Matrix(create_random_compartmental_matrix(d, p))
        
        return cls(beta, B, force_numerical=True)

    # private methods

    def _get_Qt(self, x):
        """Return matrix exponential of :math:`B` symbolically 
        if possible, or numerically.

        Args:
            x (nonnegative):
            time at which the matrix exponential is to be evaluated
            None: purely symbolic treatment
            nonnegative value: symbolic treatment at time x if possible, 
                otherwise purely numeric treatment

        Returns:
            Sympy or NumPy dxd-matrix: :math:`e^{x\\,B}

        Raises:
            Error: If purely symbolic treatment is intended by no symbolic 
                matrix exponential is available.
        """
        if x is None:
            # purely symbolic case
            if not hasattr(self, 'Qt'):
                raise(Error(
                    'No matrix exponential given for symbolic calculations'))

            Qt = self.Qt
        else:
            # numerical case
            if hasattr(self, 'Qt'):
                # if self.Qt exists, calculate symbolically and substitute t by
                # time
                t = symbols('t')
                Qt = self.Qt.subs({t: x})
            else:
                # purely numerical calculation
                B = self.B
                D = np.array([[B[i,j] for j in range(B.cols)] 
                        for i in range(B.rows)], dtype='float64') * float(x)
                Qt = expm(D)

        return Qt


    # public methods


    @property
    def beta(self): 
        """Return the initial distribution of the according Markov chain.
    
        Returns:
            SymPy or numerical dx1-matrix: 
                :math:`\\beta = \\frac{u}{\\|u\\|}`
        """
        return self.u/sum([e for e in self.u])

    @property
    def xss(self): 
        """Return the (symbolic) steady state vector.

        Returns:
            SymPy or numerical dx1-matrix: 
                :math:`x^\\ast = -B^{-1}\\,u`
        """
        return -self.B.inv()*self.u
    
    @property
    def eta(self): 
        """Return the initial distribution of the Markov chain according to 
        system age.
    
        Returns:
            SymPy or numerical dx1-matrix: 
                :math:`\\eta = \\frac{x^\\ast}{\\|x^\\ast\\|}`
        """
        return self.xss/sum([e for e in self.xss])
   
 
    # transit time

    def T_cum_dist_func(self, time=None):
        """Return the cumulative distribution function of the transit time.
    
        Args:
            time (nonnegative, optional): 
                time at which :math:`F_T` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: cumulative distribution 
            function of the transit time (evaluated at time)

        See Also:
            :func:`.phase_type.cum_dist_func`: Return the (symbolic) cumulative 
            distribution function of phase-type.
        """ 
        Qt = self._get_Qt(time)

        return phase_type.cum_dist_func(self.beta, self.B, Qt)

    def T_density(self, time=None): 
        """Return the probability density function of the transit time.
    
        Args:
            time (nonnegative, optional): 
                time at which :math:`f_T` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: probability density 
            function of the transit time (evaluated at time)

        See Also:
            :func:`.phase_type.density`: Return the (symbolic) probability 
            density function of the phase-type distribution.
        """ 
        Qt = self._get_Qt(time)

        return phase_type.density(self.beta, self.B, Qt)

    @property
    def T_expected_value(self): 
        """Return the (symbolic) expected value of the transit time.

        See Also:
            :func:`.phase_type.expected_value`: Return the (symbolic) expected 
            value of the phase-type distribution.
        """
        return phase_type.expected_value(self.beta, self.B)

    @property
    def T_standard_deviation(self): 
        """Return the (symbolic) standard deviation of the transit time.

        See Also:
            :func:`.phase_type.standard_deviation`: Return the (symbolic) 
            standard deviation of the phase-type distribution.
        """
        return phase_type.standard_deviation(self.beta, self.B)

    @property
    def T_variance(self):
        """Return the (symbolic) variance of the transit time.

        See Also:
            :func:`.phase_type.variance`: Return the (symbolic) variance of 
            the phase-type distribution.
        """
        return phase_type.variance(self.beta, self.B)

    def T_nth_moment(self, n):
        """Return the (symbolic) ``n`` th moment of the transit time.
        
        Args:
            n (positive int): order of the moment

        Returns:
            SymPy expression or numerical value: :math:`\mathbb{E}[T^n]`

        See Also:
            :func:`.phase_type.nth_moment`: Return the (symbolic) ``n`` th 
            moment of the phase-type distribution.
        """
        return phase_type.nth_moment(self.beta, self.B, n)

    def T_quantile(self, q, tol=1e-8):
        """Return a numerical quantile of the transit time distribution.
        
        The quantile is computed by a numerical inversion of the 
        cumulative distribution function.
    
        Args:
            q (between 0 and 1): probability mass to be left to the quantile
                q = 1/2 computes the median
            tol (float): tolerance for brentq algorithm of numerical 
                root search, defaults to 1e-8
    
        Returns:
            float: The smallest :math:`y` such that :math:`F_T(y)\\geq q`
    
        See Also:
            :func:`T_cum_dist_func`: Return the cumulative distribution 
            function of the transit time.
        """
        CDF = lambda t: float(self.T_cum_dist_func(t))
        try:
            res = _generalized_inverse_CDF(CDF, q, tol=tol)
        except TypeError as err:
            raise Error("Quantiles cannot be computed symbolically.") from err
            
        return res


    # system age

    def A_cum_dist_func(self, age=None): 
        """Return the cumulative distribution function of the system age.
    
        Args:
            age (nonnegative, optional): 
                age at which :math:`F_A` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: cumulative distribution 
            function of PH(:math:`\\eta`, :math:`B`)
            (evaluated at age)

        See Also:
            :func:`.phase_type.cum_dist_func`: Return the (symbolic) cumulative 
            distribution function of phase-type.
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return phase_type.cum_dist_func(self.eta, self.B, Qt).subs({t:y})
    
    def A_density(self, age=None): 
        """Return the probability density function of the system age.
    
        Args:
            age (nonnegative, optional): 
                age at which :math:`f_A` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: probability density function 
            of PH(:math:`\\eta`, :math:`B`) (evaluated at age)

        See Also:
            :func:`.phase_type.density`: Return the (symbolic) probability 
            density function of the phase-type distribution.
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return phase_type.density(self.eta, self.B, Qt).subs({t:y})
    
    @property
    def A_expected_value(self): 
        """Return the (symbolic) expected value of the system age.

        Returns:
            SymPy expression or numerical value: expected value of 
            PH(:math:`\\eta`, :math:`B`) 

        See Also:
            :obj:`.phase_type.expected_value`: Return the (symbolic) expected 
            value of the phase-type distribution.
        """
        t, y = symbols('t y')
        return phase_type.expected_value(self.eta, self.B).subs({t:y})
    
    @property
    def A_standard_deviation(self): 
        """Return the (symbolic) standard deviation of the system age.

        Returns:
            SymPy expression or numerical value: standard deviation of 
            PH(:math:`\\eta`, :math:`B`) 

        See Also:
            :func:`.phase_type.standard_deviation`: Return the (symbolic) 
            standard deviation of the phase-type distribution.
        """
        return phase_type.standard_deviation(self.eta, self.B)

    @property
    def A_variance(self):
        """Return the (symbolic) variance of the system age.

        Returns:
            SymPy expression or numerical value: variance of 
            PH(:math:`\\eta`, :math:`B`) 

        See Also:
            func:`.phase_type.variance`: Return the (symbolic) variance of the 
            phase-type distribution.
        """
        return phase_type.variance(self.eta, self.B)

    def A_nth_moment(self, n):
        """Return the (symbolic) ``n`` th moment of the system age.

        Args:
            n (positive int): order of the moment

        Returns:
            SymPy expression or numerical value: ``n`` th moment of 
            PH(:math:`\\eta`, :math:`B`) 

        See Also:
            :func:`.phase_type.nth_moment`: Return the (symbolic) ``n`` th 
            moment of the phase-type distribution.
        """
        return phase_type.nth_moment(self.eeta, self.B, n)

    def A_quantile(self, q, tol=1e-8):
        """Return a numerical quantile of the system age distribution.
        
        The quantile is computed by a numerical inversion of the 
        cumulative distribution function.
    
        Args:
            q (between 0 and 1): probability mass to be left to the quantile
                q = 1/2 computes the median
            tol (float): tolerance for brentq algorithm of numerical 
                root search, defaults to 1e-8
    
        Raises:
            Error: if attempt is made to compute quantiles symbolically

        Returns:
            float: smallest :math:`y` such that :math:`F_A(y)\\geq q`
    
        See Also:
            :func:`A_cum_dist_func`: Return the cumulative distribution 
            function of the system age.
        """
        CDF = lambda t: float(self.A_cum_dist_func(t))
        try:
            res = _generalized_inverse_CDF(CDF, q, tol=tol)
        except TypeError as err:
            raise Error("Quantiles cannot be computed symbolically.") from err
            
        return res


    # compartment age

    def a_cum_dist_func(self, age=None): 
        """Return the cumulative distribution function vector of the 
        compartment ages.
    
        Args:
            age (nonnegative, optional): 
                age at which :math:`F_a` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy or numerical dx1-matrix: :math:`F_a` (evaluated at age) 
                :math:`F_a(y) = (X^\\ast)^{-1}\\,B^{-1}\\,`
                :math:`(e^{y\\,B}-I)\\,u`
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return _age_vector_cum_dist(self.u, self.B, Qt).subs({t:y})

    def a_density(self, age=None): 
        """Return the probability density function vector of the 
        compartment ages.
    
        Args:
            age (nonnegative, optional): 
                age at which :math:`f_a` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy or numerical dx1-matrix: :math:`f_a` (evaluated at age) 
                :math:`f_a(y) = (X^\\ast)^{-1}\\,`
                :math:`e^{y\\,B}\\,u`
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return _age_vector_dens(self.u, self.B, Qt).subs({t:y})

    @property
    def a_expected_value(self): 
        """Return the (symbolic) vector of expected values of the 
        compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\mathbb{E}[a] = -(X^\\ast)^{-1}\\,`
            :math:`B^{-1}\\,x^\\ast`
        """
        t, y = symbols('t y')
        return _age_vector_exp(self.u, self.B)

    @property
    def a_standard_deviation(self): 
        """Return the (symbolic) vector of standard deviations of the 
        compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\sigma(a) = \\sqrt{\\sigma^2(a)}` component-wise
        """
        return _age_vector_sd(self.u, self.B)

    @property
    def a_variance(self):
        """Return the (symbolic) vector of variances of the compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\sigma^2(a) = \\mathbb{E}[a^2]`
            :math:`- (\\mathbb{E}[a])^2` component-wise
        """
        return _age_vector_variance(self.u, self.B)

    def a_nth_moment(self, n):
        """Return the (symbolic) vector of the ``n`` th moments of the 
        compartment ages.

        Args:
            n (positive int): order of the moment

        Returns:
            SymPy or numerical dx1-matrix:
            :math:`\\mathbb{E}[a^n] = (-1)^n\\,n!\\,(X^\\ast)^{-1}`
            :math:`\\ ,B^{-n}\\,x^\\ast`
        """
        return _age_vector_nth_moment(self.beta, self.B, n)


    def a_quantile(self, q, tol=1e-8):
        """Return a vector of numerical quantiles of the pool age distributions.
        
        The quantiles is computed by a numerical inversion of the 
        cumulative distribution functions.
    
        Args:
            q (between 0 and 1): probability mass to be left to the quantile
                q = 1/2 computes the median
            tol (float): tolerance for brentq algorithm of numerical 
                root search, defaults to 1e-8
    
        Returns:
            numpy.array: vector :math:`y=(y_1,\\ldots,y_d)` with 
            :math:`y_i` smallest value such that :math:`F_{a_i}(y_i)\\geq q`
    
        See Also:
            :func:`a_cum_dist_func`: Return the cumulative distribution function
            vector of the compartment ages.
        """
        d = self.B.rows
        res = np.nan * np.zeros(d)
        try:
            for pool in range(d):
                CDF = lambda t: float(self.a_cum_dist_func(t)[pool])
                res[pool] = _generalized_inverse_CDF(CDF, q, tol=tol)
        except TypeError as err:
            raise Error("Quantiles cannot be computed symbolically.") from err
            
        return res


    # Laplacians for T and A

    @property
    def T_laplace(self): 
        """Return the symbolic Laplacian of the transit time.

        Returns:
            SymPy expression: Laplace transform of the probability density of 
            PH(:math:`\\beta`, :math:`B`) 

        See Also:
            :obj:`.phase_type.laplace`: Return the symbolic Laplacian of the 
            phase-type distribtion.
        """
        return simplify(phase_type.laplace(self.beta, self.B))
    
    @property
    def A_laplace(self): 
        """Return the symbolic Laplacian of the system age.

        Returns:
            SymPy expression: Laplace transform of the probability density of 
            PH(:math:`\\eta`, :math:`B`) 

        See Also:
            :obj:`.phase_type.laplace`: Return the symbolic Laplacian of the 
            phase-type distribtion.
        """
        return simplify(phase_type.laplace(self.eta, self.B))


    # release

    @property
    def r_compartments(self):
        """Return the (symbolic) release vector of the system in steady state.

        Returns:
            SymPy or numerical dx1-matrix: :math:`r_j = z_j \\, x^\\ast_j`

        See Also:
            | :func:`.phase_type.z`: Return the (symbolic) vector of rates 
                toward absorbing state.
            | :obj:`xss`: Return the (symbolic) steady state vector.
        """
        r = phase_type.z(self.B)
        return Matrix(self.xss.rows, 1, [phase_type.z(self.B)[i]*self.xss[i] 
                                            for i in range(self.xss.rows)])

    @property
    def r_total(self):
        """Return the (symbolic) total system release in steady state.

        Returns:
            SymPy expression or numerical value: 
            :math:`r = \\sum\\limits_{j=1}^d r_j`

        See Also:
            :obj:`r_compartments`: Return the (symbolic) release vector of the
            system in steady state.
        """
        return sum(self.r_compartments)
      
    @property
    def absorbing_jump_chain(self):
        """Return the absorbing jump chain as a discrete-time Markov chain.

        The generator of the absorbing chain is just given by :math:`B`, which
        allows the computation of the transition probability matrix :math:`P` 
        from :math:`B=(P-I)\\,D` with :math:`D` being the diagonal matrix with
        diagonal entries taken from :math:`-B`.

        Returns:
            :class:`~.DTMC.DTMC`: :class:`DTMC` (beta, P)
        """
        # B = (P - I) * D
        d = self.B.rows
        D = diag(*[-self.B[j,j] for j in range(d)])
        P = self.B * D**(-1) + eye(d)
    
        return DTMC(self.beta, P)

    @property
    def ergodic_jump_chain(self):
        """Return the ergodic jump chain as a discrete-time Markov chain.

        The generator is given by

        .. math::
            Q = \\begin{pmatrix} 
                    B & \\beta \\\\ 
                    z^T & -1 
                \\end{pmatrix}

        and the corresponding transition probability matrix :math:`P_Q` can 
        then be obtained from :math:`Q=(P_Q-I)\\,D_Q`, where :math:`D_Q` is the 
        diagonal matrix with entries from the diagonal of :math:`-Q`.

        Returns:
           :class:`~.DTMC.DTMC`: :class:`DTMC` (beta_ext, Q) with 
           beta_ext = (beta, 0)
        """
        B = self.B
        d = B.rows
    
        lor = []
        for i in range(d):
            row = list(B[i,:]) + [self.beta[i]]
            lor.append(row)
    
        row = list(phase_type.z(B)) + [-1]
        lor.append(row)
    
        # (d+1)x(d+1) matrix
        # B     beta
        # z^T     -1
        Q = lor
        
        P = Matrix(Q)
        for j in range(d+1):
            for i in range(d+1):
                if Q[j][j] != 0:
                    if i != j:
                        P[i,j] = -Q[i][j]/Q[j][j]
                    else:
                        P[i,j] = 0
                else:
                    if i != j:
                        P[i,j] = 0
                    else:
                        P[i,j] = 1

        beta = Matrix(d+1, 1, list(self.beta) + [0])
        return DTMC(beta, P)

    #fixme: to be tested
    @property
    def entropy_per_jump(self):
        """Return the entropy per jump.
        
        Returns:
            SymPy expression or float: :math:`\\theta_J=`
            :math:`\\sum\\limits_{j=1}^{d+1} \\pi_j` 
            :math:`\\sum\\limits_{i=1}^{d+1}-p_{ij}\\,\\log p_{ij}`
            :math:`+\\sum\\limits_{j=1}^d \\pi_j\\,(1-\\log -b_{jj})` 
            :math:`+\\pi_{d+1}\\,\\sum\\limits_{i=1}^d`
            :math:`-\\beta_i\\,\\log\\beta_i`

        Notes:
            - :math:`\\pi` is the stationary distribution of the ergodic 
              jump chain.
            - :math:`\\theta_J=` entropy of ergodic jump chain + 
              entropy of sojourn times (no stay in environmental
              compartment :math:`d+1`)

        See Also:
            :obj:`~.DTMC.DTMC.stationary_distribution`: 
            Return the (symbolic) stationary distribution.
        """
        d = self.B.rows
        P = self.ergodic_jump_chain.P
        pi = self.ergodic_jump_chain.stationary_distribution
    
        theta_jumps = self.ergodic_jump_chain.ergodic_entropy

        theta_stays = 0
        for j in range(d+1):
            x = 0
    
            if j == d:
                # no entropy for stay in environment
                x += 0
            else:
                # differential entropy of exponential distribution
                x += (1-log(-self.B[j,j]))
            x *= pi[j]
            theta_stays += x

        return theta_jumps + theta_stays

    #fixme: to be tested
    @property
    def entropy_per_cycle(self):
        """Return the entropy per cycle.

        Returns:
            SymPy expression or float: entropy per jump 
            :math:`\\times` expected number of jumps per cycle

        See Also:
            | :obj:`entropy_per_jump`: Return the entropy per jump.
            | :obj:`~.DTMC.DTMC.expected_number_of_jumps`:
              Return the (symbolic) expected number of jumps before absorption.
        """
        theta_per_jump = self.entropy_per_jump
        jumps = self.absorbing_jump_chain.expected_number_of_jumps
        # add one jump from the environment into the system
        return theta_per_jump * (jumps+1)

    #fixme: to be tested
    @property
    def entropy_rate(self):
        """Return the entropy rate (entropy per unit time).

        Returns:
            SymPy expression or float: entropy per cycle 
            :math:`\\cdot\\frac{1}{\\mathbb{E}T}`

        See Also:
            | :obj:`entropy_per_cycle`: Return the entropy per cylce.
            | :obj:`T_expected_value`: Return the (symbolic) expected 
              value of the transit time.
        """
        # the entropy rate is the entropy per unit time
        # thus the entropy per cycle over cycle length
        theta_cycle = self.entropy_per_cycle
        return theta_cycle/self.T_expected_value
        
        

