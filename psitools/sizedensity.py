#!/usr/bin/python

import scipy.integrate as integrate


class SizeDensity():
    """Size density class, initialized from a function sigma that gives the size density between a minimum and maximum size. Calling the class instance with a value x < 1 returns amax*sigma(amax*x)/rho_d.

    Args:
        sigma (callable): size density function
        size_range: size range of the size density function
    """
    def __init__(self, sigma, size_range, sigma_integral=None):
        # sigma = sigma(a) between amin and amax
        self.amin = size_range[0]
        self.amax = size_range[1]

        if sigma_integral is None:
            self.rhod = integrate.quad(sigma, self.amin, self.amax)[0]
        else:
            self.rhod = sigma_integral
        self.f = lambda x: self.amax*sigma(self.amax*x)/self.rhod

        # Empty list of poles
        self.poles = []

    def __call__(self, x):
        """Return amax*sigma(amax*x)/rho_d"""
        return self.f(x)
