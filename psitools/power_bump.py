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
        return Fnn(a)*(a**3)

    mnorm = scipy.integrate.quad(mnn, amin, aR, epsrel=1e-12, epsabs=1e-12)[0]

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
