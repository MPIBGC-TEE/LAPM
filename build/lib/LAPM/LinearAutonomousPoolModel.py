"""Module for linear autonomous pool models."""
from __future__ import division

from sympy import symbols, Matrix, exp, ones, diag, simplify, eye
from math import factorial, sqrt
from numpy import array
from scipy.linalg import expm

# import the phase-type distribution
from . import PH

#############################################
# Linear autonomous compartment model class #
# d/dt x(t) = Ax(t) + u                     #
#############################################


def _age_vector_dens(u, A, Qt):
    """Return the (symbolic) probability density vector of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        A (SymPy dxd-matrix): compartment matrix
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,\\bf{A}}`

    Returns:
        SymPy dx1-matrix: probability density vector of the compartment ages
            :math:`\\bf{f_a}(y) = (\\bf{X^\\ast})^{-1}\\,e^{y\\,\\bf{A}}\\,\\bf{u}`
    """
    xss = -(A**-1)*u
    X = diag(*xss)

    return (X**-1)*Qt*u

def _age_vector_nth_moment(u, A, n):
    """Return the (symbolic) vector of nth moments of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        A (SymPy dxd-matrix): compartment matrix
        n (positive int): order of the moment

    Returns:
        SymPy dx1-matrix: vector of nth moments of the compartment ages
            :math:`\\mathbb{E}[\\bf{a}^n] = (-1)^n\\,n!\\,(\\bf{X^\\ast})^{-1}\\,\\bf{A}^{-n}\\,\\bf{x^\\ast}`
    
    See Also:
        :func:`_age_vector_exp`: Return the (symbolic) vector of expected values of the compartment ages.
    """
    xss = -(A**-1)*u
    X = diag(*xss)

    return (-1)**n*factorial(n)*(X**-1)*(A**-n)*xss

def _age_vector_exp(u, A):
    """Return the (symbolic) vector of expected values of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        A (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of expected values of the compartment ages
            :math:`\\mathbb{E}[\\bf{a}] = -(\\bf{X^\\ast})^{-1}\\,\\bf{A}^{-1}\\,\\bf{x^\\ast}`
    
    See Also:
        :func:`_age_vector_nth_moment`: Return the (symbolic) vector of nth moments of the compartment ages.
    """
    return _age_vector_nth_moment(u, A, 1)

def _age_vector_variance(u, A):
    """Return the (symbolic) vector of variances of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        A (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of variances of the compartment ages
            :math:`\\sigma^2(\\bf{a}) = \\mathbb{E}[\\bf{a}^2] - (\\mathbb{E}[\\bf{a}])^2` component-wise
    
    See Also:
        | :func:`_age_vector_exp`: Return the (symbolic) vector of expected values of the compartment ages.
        | :func:`_age_vector_nth_moment`: Return the (symbolic) vector of nth moments of the compartment ages.
    """
    sv = _age_vector_nth_moment(u, A, 2)
    ev = _age_vector_exp(u, A)
    vl = [sv[i] - ev[i]**2 for i in range(sv.rows)]
    
    return Matrix(sv.rows, 1, vl)
    
def _age_vector_sd(u, A):
    """Return the (symbolic) vector of standard deviations of the compartment ages.

    Args:
        u (SymPy dx1-matrix): external input vector
        A (SymPy dxd-matrix): compartment matrix

    Returns:
        SymPy dx1-matrix: vector of standard deviations of the compartment ages
            :math:`\\sigma(\\bf{a}) = \\sqrt{\\sigma^2(\\bf{a})}` component-wise
    
    See Also:
        :func:`_age_vector_variance`: Return the (symbolic) vector of variances of the compartment ages.
    """
    sdl = [sqrt(e) for e in _age_vector_variance(u, A)]
    
    return Matrix(len(sdl), 1, sdl)


############################################################################


class LinearAutonomousPoolModel():
    """General class of linear autonomous compartment models.

    :math:`\\frac{d}{dt} \\bf{x}(t) = \\bf{A}\\,\\bf{x}(t) + \\bf{u}`

    Notes:
        - symbolic matrix exponential Qt = :math:`e^{tA}` cannot be computed automatically
        - for symbolic computations it has to be given manually: model.Qt = ...
        - otherwise only numerical computations possible

    Attributes:
        A (SymPy dxd-matrix): The model's compartment matrix.
        u (SymPy dx1-matrix): The model's external input vector.
        Qt (SymPy dxd-matrix): Qt = :math:`e^{t\\,\\bf{A}}`                    
    """

    def __init__(self, u, A):
        """Return a linear autonomous compartment model with external inputs u and compartment matrix A.

        Args:
            u (SymPy dx1-matrix): external input vector
            A (SymPy dxd-matrix): compartment matrix
        """
        # cover one-dimensional input
        if (not hasattr(u, 'is_Matrix')) or (not u.is_Matrix): u = Matrix(1, 1, [u])
        if (not hasattr(A, 'is_Matrix')) or (not A.is_Matrix): A = Matrix(1, 1, [A])
    
        self.A = A
        self.u = u  

        # compute matrix exponential if no symbols are involved
        if A.is_Matrix and len(A.free_symbols) == 0:
            t = symbols('t')
            self.Qt = exp(t*A)

    # private methods

    def _get_Qt(self, x):
        """Return matrix exponential of A symbolically if possible or numerically.

        Args:
            x (nonnegative):
            time at which the matrix exponential is to be evaluated
            None: purely symbolic treatment
            nonnegative value: symbolic treatment at time x if possible, otherwise purely numeric treatment

        Returns:
            Sympy or NumPy dxd-matrix: :math:`e^{x\\,\\bf{A}}

        Raises:
            Exception: If purely symbolic treatment is intended by no symbolic matrix exponential is available.
        """
        if x == None:
            # purely symbolic case
            if not hasattr(self, 'Qt'):
                raise(Exception('No matrix exponential given for symbolic calculations'))

            Qt = self.Qt
        else:
            # numerical case
            if hasattr(self, 'Qt'):
                # if self.Qt exists, calculate symbolically and substitute t by time
                t = symbols('t')
                Qt = self.Qt.subs({t: x})
            else:
                # purely numerical calculation
                A = self.A
                D = array([[A[i,j] for j in range(A.cols)] for i in range(A.rows)], dtype='float32') * float(x)
                Qt = expm(D)

        return Qt


    # public methods

    @property
    def beta(self): 
        """Return the initial distribution of the according Markov chain.
    
        Returns:
            SymPy or numerical dx1-matrix: :math:`\\bf{\\beta} = \\frac{\\bf{u}}{\\|\\bf{u}\\|}`
        """
        return self.u/sum([e for e in self.u])

    @property
    def xss(self): 
        """Return the (symbolic) steady state vector.

        Returns:
            SymPy or numerical dx1-matrix: :math:`\\bf{x^\\ast} = -\\bf{A}^{-1}\\,\\bf{u}`
        """
        return -(self.A**-1)*self.u
    
    @property
    def eta(self): 
        """Return the initial distribution of the Markov chain according to system age.
    
        Returns:
            SymPy or numerical dx1-matrix: :math:`\\bf{\\eta} = \\frac{\\bf{x^\\ast}}{\\|\\bf{x^\\ast}\\|}`
        """
        return self.xss/sum([e for e in self.xss])
   
 
    # transit time
    def T_cum_dist_func(self, time = None):
        """Return the cumulative distribution function of the transit time.
    
        Args:
            time (nonnegative, optional): time at which :math:`F_T` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: cumulative distribution function of the transit time (evaluated at time)

        See Also:
            :func:`.PH.cum_dist_func`
        """ 
        Qt = self._get_Qt(time)

        return PH.cum_dist_func(self.beta, self.A, Qt)

    def T_density(self, time = None): 
        """Return the probability density of the transit time.
    
        Args:
            time (nonnegative, optional): time at which :math:`f_T` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: probability density of the transit time (evaluated at time)

        See Also:
            :func:`.PH.density`
        """ 
        Qt = self._get_Qt(time)

        return PH.density(self.beta, self.A, Qt)

    @property
    def T_expected_value(self): 
        """Return the (symbolic) expected value of the transit time.

        See Also:
            :func:`.PH.expected_value`
        """
        return PH.expected_value(self.beta, self.A)

    @property
    def T_standard_deviation(self): 
        """Return the (symbolic) standard deviation of the transit time.

        See Also:
            :func:`.PH.standard_deviation`
        """
        return PH.standard_deviation(self.beta, self.A)

    @property
    def T_variance(self):
        """Return the (symbolic) variance of the transit time.

        See Also:
            :func:`.PH.variance`
        """
        return PH.variance(self.beta, self.A)

    def T_nth_moment(self, n):
        """Return the (symbolic) nth moment of the transit time.
        
        Args:
            n (positive int): order of the moment

        Returns:
            SymPy expression or numerical value: :math:`\mathbb{E}[T^n]`

        See Also:
            :func:`.PH.nth_moment`
        """
        return PH.nth_moment(self.beta, self.A, n)


    # system age
    def A_cum_dist_func(self, age = None): 
        """Return the cumulative distribution function of the system age.
    
        Args:
            age (nonnegative, optional): age at which :math:`F_A` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: cumulative distribution function of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 
            (evaluated at age)

        See Also:
            :func:`.PH.cum_dist_func`
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return PH.cum_dist_func(self.eta, self.A, Qt).subs({t:y})
    
    def A_density(self, age = None): 
        """Return the probability density of the system age.
    
        Args:
            age (nonnegative, optional): age at which :math:`f_A` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy expression or numerical value: probability density of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 
            (evaluated at age)

        See Also:
            :func:`.PH.density`
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return PH.density(self.eta, self.A, Qt).subs({t:y})
    
    @property
    def A_expected_value(self): 
        """Return the (symbolic) expected value of the system age.

        Returns:
            SymPy expression or numerical value: expected value of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 

        See Also:
            :func:`.PH.expected_value`
        """
        t, y = symbols('t y')
        return PH.expected_value(self.eta, self.A).subs({t:y})
    
    @property
    def A_standard_deviation(self): 
        """Return the (symbolic) standard deviation of the system age.

        Returns:
            SymPy expression or numerical value: standard deviation of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 

        See Also:
            :func:`.PH.standard_deviation`
        
        """
        return PH.standard_deviation(self.eta, self.A)

    @property
    def A_variance(self):
        """Return the (symbolic) variance of the system age.

        Returns:
            SymPy expression or numerical value: variance of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 

        See Also:
            :func:`.PH.variance`

        """
        return PH.variance(self.eta, self.A)

    def A_nth_moment(self, n):
        """Return the (symbolic) nth moment of the system age.

        Args:
            n (positive int): order of the moment

        Returns:
            SymPy expression or numerical value: nth moment of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 

        See Also:
            :func:`.PH.nth_moment`
        
        """
        return PH.nth_moment(self.eeta, self.A, n)


    # compartment age
    def a_density(self, age = None): 
        """Return the probability density vector of the compartment ages.
    
        Args:
            age (nonnegative, optional): age at which :math:`\\bf{f_a}` is to be evaluated,
                defaults to None: purely symbolic treatment
    
        Returns:
            SymPy or numerical dx1-matrix: :math:`f_a` (evaluated at age) 
                :math:`\\bf{f_a}(y) = (\\bf{X^\\ast})^{-1}\\,e^{y\\,\\bf{A}}\\,\\bf{u}`
        """ 
        Qt = self._get_Qt(age)
        t, y = symbols('t y')
        return _age_vector_dens(self.u, self.A, Qt).subs({t:y})

    @property
    def a_expected_value(self): 
        """Return the (symbolic) vector of expected values of the compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\mathbb{E}[\\bf{a}] = -(\\bf{X^\\ast})^{-1}\\,\\bf{A}^{-1}\\,\\bf{x^\\ast}`
        """
        t, y = symbols('t y')
        return _age_vector_exp(self.u, self.A)

    @property
    def a_standard_deviation(self): 
        """Return the (symbolic) vector of standard deviations of the compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\sigma(\\bf{a}) = \\sqrt{\\sigma^2(\\bf{a})}` component-wise
        """
        return _age_vector_sd(self.u, self.A)

    @property
    def a_variance(self):
        """Return the (symbolic) vector of variances of the compartment ages.

        Returns:
            SymPy dx1-matrix:
            :math:`\\sigma^2(\\bf{a}) = \\mathbb{E}[\\bf{a}^2] - (\\mathbb{E}[\\bf{a}])^2` component-wise
        """
        return _age_vector_variance(self.u, self.A)

    def a_nth_moment(self, n):
        """Return the (symbolic) vector of the nth moments of the compartment ages.

        Args:
            n (positive int): order of the moment

        Returns:
            SymPy or numerical dx1-matrix:
            :math:`\\mathbb{E}[\\bf{a}^n] = (-1)^n\\,n!\\,(\\bf{X^\\ast})^{-1}\\,\\bf{A}^{-n}\\,\\bf{x^\\ast}`
        """
        return _age_vector_nth_moment(self.beta, self.A, n)


    # Laplacians for T and A
    @property
    def T_laplace(self): 
        """Return the symbolic Laplacian of the transit time.

        See Also:
            :func:`.PH.laplace`
        """
        return simplify(PH.laplace(self.beta, self.A))
    
    @property
    def A_laplace(self): 
        """Return the symbolic Laplacian of the system age.

        Returns:
            SymPy expression: Laplace transform of the probability density of PH(:math:`\\bf{\\eta}`, :math:`\\bf{A}`) 

        See Also:
            :func:`.PH.laplace`
        """
        return simplify(PH.laplace(self.eta, self.A))


    # release
    @property
    def r_compartments(self):
        """Return the (symbolic) release vector of the system in steady state.

        Returns:
            SymPy or numerical dx1-matrix: :math:`r_j = z_j * x^\\ast_j`

        See Also:
            | :func:`.PH.z`
            | :func:`xss`
        """
        r = PH.z(self.A)
        return Matrix(self.xss.rows, 1, [PH.z(self.A)[i]*self.xss[i] for i in range(self.xss.rows)])

    @property
    def r_total(self):
        """Return the (symbolic) total system release in steady state.

        Returns:
            SymPy expression or numerical value: :math:`r = \sum_j r_j`

        See Also:
            :func:`r_compartments`
        """
        return sum(self.r_compartments)
       
 
