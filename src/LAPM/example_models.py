"""Example linear autonomous pool models."""
from __future__ import division

from sympy import symbols, Matrix, exp

from .linear_autonomous_pool_model import LinearAutonomousPoolModel


############################
# example compartment models
############################
    

class TwoPoolsNoFeedback(LinearAutonomousPoolModel):
    """Two-compartment model with no feedback.

    .. math:: 
        u = \\begin{pmatrix} 
                u_1 \\\\ 
                u_2
            \\end{pmatrix},
        \\quad
        A = \\begin{pmatrix} 
                          -\\lambda_1 &           0 \\\\
                 \\alpha\\,\\lambda_1 & -\\lambda_2
           \\end{pmatrix}

    Attributes:
        Qt (SymPy matrix): Qt = :math:`e^{t\\,A}` is given
    """

    def __init__(self, alpha, u_1, u_2):
        """Return a two-compartment model with no feedback.

        Args:
            alpha (in [0,1] or SymPy expression): 
                proportion of outflow from pool 1 that goes to pool 2
            u_1 (nonnegative or SymPy expression): external input rate to pool 1
            u_2 (nonnegative or SymPy expression): external input rate to pool 2

        Note:
            The symbolic matrix exponential will be initialized automatically.
        """
        lamda_1, lamda_2 = symbols('lamda_1 lamda_2', positive=True)
        A = Matrix([[     -lamda_1,        0],
                    [alpha*lamda_1, -lamda_2]])
        u = Matrix(2, 1, [u_1, u_2])
        super().__init__(u, A)

        # The symbolic matrix exponential cannot be calculated automatically...
        t = symbols('t')
        self.Qt = Matrix(
            [[exp(-lamda_1*t),               0],
            [alpha*lamda_1/(lamda_1-lamda_2)*(exp(-lamda_2*t)-exp(-lamda_1*t)), 
                exp(-lamda_2*t)]])

class TwoPoolsFeedbackSimple(LinearAutonomousPoolModel):
    """Two-compartment model with no feedback.

    .. math::
        u = \\begin{pmatrix}
                u_1 \\\\
                  0
            \\end{pmatrix},
        \\quad
        A = \\begin{pmatrix}
                           -\\lambda_1 &  \\lambda_2 \\\\
                  \\alpha\\,\\lambda_1 & -\\lambda_2
            \\end{pmatrix}

    Inputs and outputs only through compartment 1.
    """

    def __init__(self, alpha, u_1):
        """Return a simple two-compartment model with feedback.

        Args:
            alpha (in [0,1] or SymPy expression): 
                proportion of outflow from pool 1 that goes to pool 2
            u_1 (nonnegative or SymPy expression): external input rate to pool 1

        Note:
            The symbolic matrix exponential will NOT be initialized 
            automatically.
        """
        lamda_1, lamda_2 = symbols('lamda_1 lamda_2', positive=True)
        A = Matrix([[     -lamda_1,  lamda_2],
                    [alpha*lamda_1, -lamda_2]])
        u = Matrix(2, 1, [u_1, 0])
        super().__init__(u, A)


class TwoPoolsFeedback(LinearAutonomousPoolModel):
    """Two-compartment model with feedback.

    .. math::
        u = \\begin{pmatrix}
                 u_1 \\\\
                 u_2
            \\end{pmatrix},
        \\quad
        A = \\begin{pmatrix}
                            -\\lambda_1 & \\alpha_{12}\\,\\lambda_2 \\\\
                \\alpha_{21}\\,\\lambda_1 &             -\\lambda_2
            \\end{pmatrix}
    """

    def __init__(self, alpha_12, alpha_21, u_1, u_2):
        """Return complete two-compartment model with feedback.

        Args:
            alpha_12 (in [0,1] or SymPy expression): 
                proportion of outflow from pool 2 that goes to pool 1
            alpha_21 (in [0,1] or SymPy expression): 
                proportion of outflow from pool 1 that goes to pool 2
            u_1 (nonnegative or SymPy expression): external input rate to pool 1
            u_2 (nonnegative or SymPy expression): external input rate to pool 2

        Note:
            The symbolic matrix exponential will NOT be initialized 
                automatically.
        """
        lamda_1, lamda_2 = symbols('lamda_1 lamda_2', positive=True)
        A = Matrix([[        -lamda_1,  alpha_12*lamda_2],
                    [alpha_21*lamda_1,          -lamda_2]])
        u = Matrix(2, 1, [u_1, u_2])
        super().__init__(u, A)


