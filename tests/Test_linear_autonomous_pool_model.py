# vim:set ff=unix expandtab ts=4 sw=4:
import unittest
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys

import numpy as np
from sympy import Matrix, symbols, zeros, log

from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel, Error
from LAPM.example_models import TwoPoolsNoFeedback


class TestLinearAutonmousPoolModel(unittest.TestCase):

    def test_absorbing_jump_chain(self):
        M = TwoPoolsNoFeedback(0, 1, 1)
        jc = M.absorbing_jump_chain
        ref = Matrix(2, 1, [1/2, 1/2])
        self.assertEqual(jc.beta, ref) 
        self.assertEqual(jc.P, zeros(2,2))

    def test_ergodic_jump_chain(self):
        M = TwoPoolsNoFeedback(0, 1, 1)
        jc = M.ergodic_jump_chain
        ref = Matrix(3, 1, [1/2, 1/2, 0])
        self.assertEqual(jc.beta, ref) 
        ref = Matrix([[0, 0, 1/2], 
                      [0, 0, 1/2], 
                      [1, 1,   0]])        
        self.assertEqual(jc.P, ref)

    def test_entropy_per_jump(self):
        # one pool
        u = Matrix(1, 1, [1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = 1/2*(1-log(lamda))
        self.assertEqual(M.entropy_per_jump, ref)

        # two pools, serial
        u = Matrix(2, 1, [1, 0])
        lamda = symbols('lamda')
        B = Matrix([[-lamda,      0],
                    [ lamda, -lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = 1/3*(1-log(lamda))*2
        self.assertEqual(M.entropy_per_jump, ref)

        # two pools, parallel
        u = Matrix(2, 1, [1, 1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda,      0],
                    [     0, -lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = 1/4*(1-log(lamda))*2 + 1/2*log(2)
        self.assertEqual(M.entropy_per_jump, ref)

    def test_entropy_per_cycle(self):
        # one pool
        u = Matrix(1, 1, [1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = 1/2*(1-log(lamda))*1 * 2
        self.assertEqual(M.entropy_per_cycle, ref)

        # two pools, serial
        u = Matrix(2, 1, [1, 0])
        lamda = symbols('lamda')
        B = Matrix([[-lamda,      0],
                    [ lamda, -lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = 1/3*(1-log(lamda))*2 * 3
        self.assertEqual(M.entropy_per_cycle, ref)

        # two pools, parallel
        u = Matrix(2, 1, [1, 1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda,      0],
                    [     0, -lamda]])
        M = LinearAutonomousPoolModel(u, B)
        ref = (1/4*(1-log(lamda))*2 + 1/2*log(2)) * 2
        self.assertEqual(M.entropy_per_cycle, ref)

    def test_entropy_rate(self):
        # one pool
        u = Matrix(1, 1, [1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda]])
        M = LinearAutonomousPoolModel(u, B)
        # entropy rate of Poisson process
        ref = lamda * (1-log(lamda))
        self.assertEqual(M.entropy_rate, ref)

    def test_T_quantile(self):
        # one pool
        u = Matrix(1, 1, [1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda]])
        M = LinearAutonomousPoolModel(u, B)
        # median of general exponential distribution
        # cannot be computed here symbolically
        with self.assertRaises(Error):
            print(M.T_quantile(0.5))
        
        B = Matrix([[-1]])
        M = LinearAutonomousPoolModel(u, B)
        # median of exponential distribution with lamda = 1
        ref = np.log(2)
        self.assertTrue(np.allclose(M.T_quantile(0.5), ref))

    def test_A_quantile(self):
        # one pool
        u = Matrix(1, 1, [1])
        lamda = symbols('lamda')
        B = Matrix([[-lamda]])
        M = LinearAutonomousPoolModel(u, B)
        # median of general exponential distribution
        # cannot be computed here symbolically
        with self.assertRaises(Error):
            print(M.A_quantile(0.5))
        
        B = Matrix([[-1]])
        M = LinearAutonomousPoolModel(u, B)
        # median of exponential distribution with lamda = 1
        ref = np.log(2)
        self.assertTrue(np.allclose(M.A_quantile(0.5), ref))

    def test_a_quantile(self):
        # two pools
        u = Matrix(2, 1, [1, 1])
        B = Matrix([[-1,  0],
                    [ 0, -2]])
        M = LinearAutonomousPoolModel(u, B)
        
        # vector of medians of exponential distributions with 
        # lamda_1 = 1 and lamda_2 = 2
        ref = np.array([np.log(2), 
                        0.5*np.log(2)])
        self.assertTrue(np.allclose(M.a_quantile(0.5), ref))





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


