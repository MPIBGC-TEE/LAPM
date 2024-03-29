"""Discrete version of :module:`~LAPM.linear_autonomous_pool:model`.

"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import LinAlgError, det, pinv
from scipy.special import factorial

from LAPM import picklegzip


class DiscreteLinearAutonomousPoolModel:
    """Discrete version of :class:`~LAPM.linear_autonomous_pool_model.LinearAutonomousPoolModel`.

    :math:`x_{n+1} = B\,x_n + U`, system in equilibrium

    Time step size is fixed to 1. No symbolic treatment, purely numerical.
    """

    def __init__(self, U: np.ndarray, B: np.ndarray, check_col_sums: bool = True):
        """
        Args:
            U: one-dimensional external input vector
            B: discrete compartmental matrix

                - nonnegative entries
                - col sums not grater than 1

            check_col_sums: if False, the column sum condition will not be checked,
                can be useful in case of minimal transgressions
        """
        if (U < 0).sum() > 0:
            raise ValueError("Negative values in input vector.")

        if (B < 0).sum() > 0:
            raise ValueError("Negative values in discrete compartmental matrix.")

        if check_col_sums:
            col_sum_vector = B.sum(axis=0)
            if (col_sum_vector > 1).sum() > 0:
                raise ValueError(
                    "Column sums not bounded by 1 in discrete comp. matrix."
                )

        self.U = U
        self.B = B

        Id = np.identity(self.nr_pools)

        # test if matrix is invertible
        if det(Id - B) == 0:
            warnings.warn("Id-B is singular, working with pseudo inverse now.")

    def restrict_to_pools(
        self, pool_nrs: np.ndarray, check_col_sums: bool = True
    ) -> "__class__":
        """Restrict the discrete model run to a subset of pools.

        Args:
            pool_nrs: array of pool numbers INSIDE the resctricted model,
                all other pools will be considered as OUSTIDE
            check_col_sums: if False, the column sum condition will not be checked,
                can be useful in case of minimal transgressions

        Returns:
            a DMR in equilibrium with ``len(pool_nrs)`` pools
        """
        nr_pools = len(pool_nrs)
        U_restricted = np.nan * np.ones(nr_pools)
        U_restricted[:] = self.U[pool_nrs]

        B_restricted = np.nan * np.ones((nr_pools, nr_pools))
        B_restricted = self.B[pool_nrs][:, pool_nrs]

        dmr_eq_restricted = self.__class__(U_restricted, B_restricted, check_col_sums)
        return dmr_eq_restricted

    @property
    def nr_pools(self) -> float:
        """Number of pools of the compartmental system."""
        return len(self.U)

    @property
    def xss(self) -> np.ndarray:
        """Steady-state vector."""
        Id = np.identity(self.nr_pools)
        return pinv(Id - self.B) @ self.U

    def _factorial_moment_vector(self, order: int) -> np.ndarray:
        """Internal method, helps to compute age moments."""
        Id = np.identity(self.nr_pools)

        B = self.B
        x = self.xss
        X = x * Id
        n = order

        fm = factorial(n) * pinv(X) @ matrix_power(B, n)
        fm = fm @ matrix_power(pinv(Id - B), n) @ x

        return fm

    def age_moment_vector_up_to(self, up_to_order: int) -> np.ndarray:
        """Collection of vectors of age moments.

        Args:
            up_to_order: moments until this order will be considered;
                order=1: only the mean

        Returns:
            nd.ndarray (up_to_order x nr_pools): age moment vectors
        """

        def stirling(n, k):
            n1 = n
            k1 = k
            if n <= 0:
                return 1
            elif k <= 0:
                return 0
            elif n == 0 and k == 0:
                return -1
            elif n != 0 and n == k:
                return 1
            elif n < k:
                return 0
            else:
                temp1 = stirling(n1 - 1, k1)
                temp1 = k1 * temp1

            return (k1 * (stirling(n1 - 1, k1))) + stirling(n1 - 1, k1 - 1)

        nr_pools = self.nr_pools
        age_moments = []
        for n in range(1, up_to_order + 1):
            m_moment = np.zeros(nr_pools)
            for k in range(n + 1):
                m_moment += stirling(n, k) * self._factorial_moment_vector(k)

            age_moments.append(m_moment)

        return np.array(age_moments)

    def age_moment_vector(self, order: int) -> nd.npdarray:
        """Vector of age moment of requested ``order``."""
        return self.age_moment_vector_up_to(order)[order - 1]

    def system_age_moment(
        self, order: int, xss: np.ndarray = None, mask: np.ndarray = False
    ) -> float:
        """Moment of the age of all material in the system.

        Args:
            order: 1 = mean, 2 = variance, ...
            mask (nr_pools): boolean, pools with True values will be ignored
            xss (nr_pools): vector of pool contents to be used; usually,
                self.xss (default) is used, but when (mis-)using the class
                in order to fake equilibrium start ages for a time-dependent DMR,
                the x0 vector of that dmr should be used

        Returns:
            Moment of the system age
        """
        age_moment_vector = self.age_moment_vector(order)
        if xss is None:
            xss = self.xss

        xss = np.ma.masked_array(xss, mask)
        total_mass = xss.sum()

        return (age_moment_vector * xss).sum() / total_mass

    def age_masses_func(self, xss: np.ndarray = None) -> Callable[[int], float]:
        """Function of age index, returns mass with with this age.

        Args:
            xss (nr_pools): vector of pool contents to be used; usually,
                self.xss (default) is used, but when (mis-)using the class
                in order to fake equilibrium start ages for a time-dependent DMR,
                the x0 vector of that dmr should be used

        Returns:
            p0: function ``of`` ai (age index), returns mass with age ``ai``
        """
        Id = np.identity(self.nr_pools)
        U = self.U
        B = self.B

        def p0_self(ai):
            if ai < 0:
                return np.zeros_like(U)
            return matrix_power(B, ai) @ U  # if age zero exists

        if xss is None:
            return p0_self
        else:
            # rescale from fake equilibrium pool contents to start_vector contents
            renorm_vector = xss / self.xss
            p0 = lambda ai: p0_self_eq(ai) * renorm_vector
            return p0

    def cumulative_age_masses_func(self, xss: np.ndarray = None):
        """Function of age index, returns cumulative mass with with this age.

        Args:
            xss (nr_pools): vector of pool contents to be used; usually,
                self.xss (default) is used, but when (mis-)using the class
                in order to fake equilibrium start ages for a time-dependent DMR,
                the x0 vector of that dmr should be used

        Returns:
            P0: function ``of`` ai (age index), returns mass with age equal to
                or less than ``ai``
        """
        Id = np.identity(self.nr_pools)
        U = self.U
        B = self.B

        IdmB_inv = pinv(Id - B)

        def P0_self(ai):  # ai = age bin index
            if ai < 0:
                return np.zeros_like(U)
            return IdmB_inv @ (Id - matrix_power(B, ai + 1)) @ U

        if xss is None:
            return P0_self
        else:
            # rescale from fake equilibrium pool contents to start_vector contents
            renorm_vector = xss / self.xss
            P0 = lambda ai: P0_self(ai) * renorm_vector
            return P0

    def external_output_rate_vector(self):
        """Vector of external output rates."""
        rho = 1 - self.B.sum(0)
        return rho

    def external_output_vector(self):
        """Vector of external outputs."""
        rho = self.external_output_rate_vector()
        r = rho * self.xss
        return r

    def transit_time_moment(self, order):
        """Moment of the age of the material in the outflux.

        Args:
            order: 1 = mean, 2 = variance, ...

        Returns:
            Moment of the transit time
        """
        age_moment_vector = self.age_moment_vector(order)
        r = self.external_output_vector()
        tt_moment = (r * age_moment_vector).sum() / r.sum()
        return tt_moment

    def transit_time_mass_func(self):
        """Function of age index, returns mass in the outflux with with this age.

        Returns:
            p0_tt: function ``of`` ai (age index), returns mass with age ``ai``
                in the outflux
        """
        p0 = self.age_masses_func()
        rho = self.external_output_rate_vector()

        def p0_tt(ai):
            return (rho * p0(ai)).sum()

        return p0_tt

    #### disk operations ####

    @classmethod
    def load_from_file(cls, filename):
        """Load a class instance from disk."""
        dmr_eq = picklegzip.load(filename)
        return dmr_eq

    def save_to_file(self, filename):
        """Save a class instance to disk."""
        picklegzip.dump(self, filename)
