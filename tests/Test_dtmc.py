# vim:set ff=unix expandtab ts=4 sw=4:
import unittest
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys

from sympy import Matrix, symbols, log

from LAPM.dtmc import DTMC, Error

class TestDTMC(unittest.TestCase):

    def test_fundamental_matrix(self):
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1/2, 1/2])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        N = mc.fundamental_matrix
        ref = Matrix(
            [[   -1/(p_12*p_21 - 1), -p_12/(p_12*p_21 - 1)],
             [-p_21/(p_12*p_21 - 1),   - 1/(p_12*p_21 - 1)]])
                
        self.assertEqual(N, ref)

        # two pools, infinite loop
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1, 0])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 1, p_12: 1})

        with self.assertRaises(Error):
            print(mc.fundamental_matrix)

    def test_expected_number_of_jumps(self):
        # two pools, parallel
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1/2, 1/2])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 0, p_12: 0})
        self.assertEqual(mc.expected_number_of_jumps, 1) 
    
        # two pools, serial
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1, 0])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 1, p_12: 0})
        self.assertEqual(mc.expected_number_of_jumps, 2) 

        # two pools, infinite loop
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1, 0])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 1, p_12: 1})

        with self.assertRaises(Error):
            print(mc.expected_number_of_jumps)

    def test_stationary_distribution(self):
        # two pools, parallel, empty system, no stationary distribution
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1/2, 1/2])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 0, p_12: 0})
        pi = mc.stationary_distribution
        self.assertEqual(pi, None) 
    
        # two pools, parallel, closed
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1/2, 1/2])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 1, p_12: 1})
        pi = mc.stationary_distribution
        ref = Matrix(2, 1, [1/2, 1/2])
        self.assertEqual(pi, ref)

    def test_ergodic_entropy(self):
        # two pools, parallel, closed
        p_21, p_12 = symbols('p_21 p_12', positive=True)
        beta = Matrix(2, 1, [1/2, 1/2])
        P = Matrix([[   0, p_12],
                    [p_21,    0]])
        mc = DTMC(beta, P)
        mc.P = mc.P.subs({p_21: 1, p_12: 1})
        self.assertEqual(mc.ergodic_entropy, 0)
     
        # two pools parallel plus environment
        beta = Matrix(3, 1, [1/2, 1/2, 0])
        P = Matrix([[0, 0, 1/2],
                    [0, 0, 1/2],
                    [1, 1,   0]])
        mc = DTMC(beta, P)
        self.assertEqual(mc.ergodic_entropy, float(1/2*log(2)))
     
    
################################################################################


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(".",pattern=__file__)
    # Run same tests across 16 processes
    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(16))
    runner = unittest.TextTestRunner()
    res=runner.run(concurrent_suite)
    # to let the buildbot fail we set the exit value !=0 if either a failure or 
    # error occurs
    if (len(res.errors)+len(res.failures))>0:
        sys.exit(1)


