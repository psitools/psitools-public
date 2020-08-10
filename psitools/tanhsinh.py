#!/usr/bin/python

import numpy as np
import warnings


class TanhSinh():
    """Basic implementation of tanh-sinh quadrature. When creating an instance, all abscissae and weights are calculated and can then be used on multiple integrals. No error estimate is provided; for integrals without endpoint singularities it should be close to 10^(-precision_digit).

    Args:
        precision_digit (optional): precision level in digits. Defaults to 15, close to double precision machine limit.
        max_level (optional): maximum level (minimum step size is 2^(-max_level)). Defaults to 12.
    """
    def __init__(self, precision_digit=15, max_level=12):
        # Precision level in digits
        self.p = precision_digit
        # Maximum level
        self.max_m = max_level
        # Maximum allowed precision
        self.eps = np.power(10.0, -self.p)

        # Minimum step size
        h_min = np.power(2.0, -self.max_m)

        # Estimate the maximum number of terms needed
        Nmax = int(np.arcsinh((2.0/np.pi)*np.arctanh(1 - self.eps))/h_min) + 10

        hj = h_min*np.arange(Nmax)
        sinhj = 0.5*np.pi*np.sinh(hj)
        coshj = 0.5*np.pi*np.cosh(hj)
        coshsinh = np.cosh(sinhj)

        # Abscissae and weights
        x = 1.0 - 1.0/(np.exp(sinhj)*coshsinh)
        w = coshj/(coshsinh*coshsinh)

        # Make sure we are not evaluating exactly at the endpoints
        cond1 = np.asarray(x < 1.0 - self.eps)
        cond2 = np.asarray(w > self.eps)

        # Select only valid abscissae and weights
        sel = np.logical_and(cond1, cond2).nonzero()

        # Final abscissa and weights
        self.xj = x[sel]
        self.wj = coshj[sel]/(coshsinh[sel]*coshsinh[sel])

    def t(self, x, a, b):
        """Scaling of interval to -1,1"""
        return 0.5*(x*(b - a) + b + a)

    def integrate(self, func, a, b, tol=None):
        """Integrate function between a and b using tanh-sinh quadrature.

        Args:
            func: callable, function f(x) to be integrated
            a: lower integration bound
            b: upper integration bound
            tol (optional): required tolerance. If set to None, try to achieve maximum precision.
        """
        # By default, go to maximum precision
        if tol is None:
            tol = self.eps
        else:
            if tol < self.eps:
                warnings.warn(
                  'TanhSinh: Requested tolerance too small, setting tol = {:e}'
                  .format(self.eps))
                tol = self.eps

        # Transform to interval -1, 1
        self.f = lambda x: 0.5*(b - a)*func(self.t(x, a, b))

        # Make sure we do not reach the end points
        sel = np.asarray(self.t(self.xj, a, b) < b - self.eps).nonzero()

        xj = self.xj[sel]
        wj = self.wj[sel]

        # First guess at h=1
        step = np.power(2, self.max_m)
        result = \
          np.sum(wj[::step]*self.f(xj[::step])) + \
          np.sum(wj[step::step]*self.f(-xj[step::step]))

        # Decrease h until tolerance reached
        for m in range(1, self.max_m + 1):
            h = np.power(2.0, -m)
            step = np.power(2, self.max_m - m)

            # All weights and abscissae to be used
            w = wj[::step]
            x = xj[::step]

            # Change in result involves only odd entries
            dresult = -0.5*result + \
              h*np.sum(w[1::2]*self.f(x[1::2])) + \
              h*np.sum(w[1::2]*self.f(-x[1::2]))

            #print(m, dresult, tol)

            # Done when tolerance reached
            if np.abs(dresult) < tol:
                break

            result = result + dresult

        return result


class TanhSinhNoDeepCopy(TanhSinh):
    """ This subclass exists because of the use of deepcopy in psi_grid_refine
    and other parallel drivers. As for most scans the TanhSinh object is
    exactly the same for all points, using this class os the prototype argument
    will prevent memory being usedup with deep copies of the exact same thing.
    """

    def __deepcopy__(self, memo):
        """ This implementation eliminates the ability of copy.deepcopy to
        deep copy the instance, instead returning a reference."""
        return self
