#
# Module to contain definitions for the Powerlaw + Bump Birnstiel distribution
#

import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.integrate
import scipy.optimize

# The function is broken up into a core and two wrappers, as different aspects
# are needed at different times.


def lognormpdf(y, s, loc, scale=1.0):
    """Avoid actually using scipy.
    """
    x = (y - loc) / scale
    return np.exp(-np.log(x)**2 / (2*s**2) - np.log(s*x*np.sqrt(2.0*np.pi)))


def _handle_args_core(amin, aP, aL, aR, bumpfac, beta):
    """ Common core for the two interfaces. """
    # Defaults are functions of aP
    if aL is None:
        aL = 2.0/3.0*aP
    if aR is None:
        aR = 1.56*aP

    fnn = lambda a: a**(beta)/aP**(beta)

    sigma = np.max((np.min((np.abs(aR-aP), np.abs(aL-aP)))
                   / np.sqrt(np.log(2.0)), 0.1*aP))

    # The original formula has a Gaussian bump
    # b = lambda a: 2.0*fnn(aL)*np.exp(-(a-aP)**2/sigma**2)
    # Replace with a lognormal bump, as that's basically the same
    # and more like aerodynamic sorting
    b = lambda a: bumpfac*fnn(aL)*np.exp(-np.log(a-(-1+aP))**2/sigma**2)

    return aL, aR, fnn, sigma, b


def get_sigma0_birnstiel_bump(amin, aP, epstot, aL=None, aR=None,
                              bumpfac=2.0, beta=-3.5):
    """ A normalized dust distribution inspired by the recipe in
        Birnstiel et al 2011 2011A&A...525A..11B

    amin : minimum monomer size
    aL: left edge of cratering bump
    aP: peak of cratering bump
    aR: max particle size
    bumpfac: factor for bump over power law peak, 2.0 is Birnstiel model

    Default argument values correspond to MRN + a bump which the
    Birnstiel et al 2011 generates when givin inner MMSN midplane-like values.

    Birnstiel, T., Ormel, C. W., & Dullemond, C. P. (2011).
    Dust size distributions in coagulation/fragmentation equilibrium:
    numerical solutions and analytical fits. Astronomy and Astrophysics,
    525, 11. http://doi.org/10.1051/0004-6361/201015228
    """
    aL, aR, fnn, sigma, b = _handle_args_core(amin, aP, aL, aR, bumpfac, beta)

    # A wrapper to make the whole thing handle array arguments
    def Fnn_vec(a):
        a = np.asarray(a)
        scalar_input = False
        if a.ndim == 0:
            a = a[None]  # Makes x 1D
            scalar_input = True
        f = np.vectorize(Fnn, otypes='f')(a)
        if scalar_input:
            return np.squeeze(f)
        return f

    def Fnn(a):
        if a < amin:
            return 0.0
        elif a <= aL:
            return fnn(a)
        elif a <= aP:
            return np.max((fnn(a), b(a)))
        elif a <= aR:
            return b(a)
        else:
            return 0.0

    def mnn(a):
        return Fnn_vec(a)*(a**3)

    mnorm = scipy.integrate.quad(mnn, amin, aR, epsrel=1e-12, limit=100)[0]

    sigma0 = lambda a: mnn(a)/mnorm*epstot
    return sigma0


# Refactor this, has a bunch of repeated code.
def get_birnstiel_discontinuity(amin, aP, aL=None, aR=None,
                                bumpfac=2.0, beta=-3.5):
    """ Find the transition point in the form above.
    """

    aL, aR, fnn, sigma, b = _handle_args_core(amin, aP, aL, aR, bumpfac, beta)

    # Need to solve fnn = b between aL and aP
    f = lambda x: fnn(x) - b(x)
    return scipy.optimize.bisect(f, aL, aP)


class PowerBump():
    def __init__(self, amin, aP, aL=None, aR=None, bumpfac=2.0, beta=-3.5):
        self.amin = amin
        self.aP = aP
        if aL is None:
            self.aL = 2*aP/3
        else:
            self.aL = aL
        if aR is None:
            self.aR = 1.56*aP
        else:
            self.aR = aR

        sigma = np.max((np.min((np.abs(self.aR-aP), np.abs(self.aL-aP)))
                       / np.sqrt(np.log(2.0)), 0.1*aP))

        fac_fnn = aP**(-beta)
        self.fnn = lambda a: fac_fnn*a**(beta)

        fac_b = bumpfac*self.fnn(self.aL)
        fac_b2 = -1/sigma**2
        self.b = lambda a: fac_b*np.exp(fac_b2*np.log(a-(-1+aP))**2)

    def get_discontinuity(self):
        # Need to solve fnn = b between aL and aP
        f = lambda x: self.fnn(x) - self.b(x)
        return scipy.optimize.bisect(f, self.aL, self.aP)

    def sigma0(self, a):
        # Make sure we can handle both vector and scalar a
        a = np.asarray(a)
        scalar_input = False
        if a.ndim == 0:
            a = a[None]  # Makes w 1D
            scalar_input = True
        else:
            original_shape = np.shape(a)
            a = np.ravel(a)

        Fnn = self.fnn(a)
        bb = self.b(a)

        maxbf = np.maximum(Fnn, bb)

        sel = np.asarray(a > self.aL).nonzero()
        Fnn[sel] = maxbf[sel]

        sel = np.asarray(a > self.aP).nonzero()
        Fnn[sel] = bb[sel]

        ret = Fnn*a**3

        # Return value of original shape
        if scalar_input:
            return np.squeeze(ret)
        return np.reshape(ret, original_shape)
