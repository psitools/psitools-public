#
# Module to contain definitions for the Powerlaw + Bump Birnstiel distribution
#

import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.integrate
import scipy.optimize


def lognormpdf(y, s, loc, scale=1.0):
    """Avoid actually using scipy.
    """
    x = (y - loc) / scale
    return np.exp(-np.log(x)**2 / (2*s**2) - np.log(s*x*np.sqrt(2.0*np.pi)))


class PowerBump():
    def __init__(self, amin, aP, aL=None, aR=None, bumpfac=2.0, beta=-3.5,
                 epstot=None):
        """ A normalized dust distribution inspired by the recipe in
            Birnstiel et al 2011 2011A&A...525A..11B

        amin : minimum monomer size
        aL: left edge of cratering bump
        aP: peak of cratering bump
        aR: max particle size
        bumpfac: factor for bump over power law peak, 2.0 is Birnstiel model
        epstot: total dust mass fraction to normalize (optional)

        Default argument values correspond to MRN + a bump which the
        Birnstiel et al 2011 generates when givin inner MMSN midplane-like
        values.

        Birnstiel, T., Ormel, C. W., & Dullemond, C. P. (2011).
        Dust size distributions in coagulation/fragmentation equilibrium:
        numerical solutions and analytical fits. Astronomy and Astrophysics,
        525, 11. http://doi.org/10.1051/0004-6361/201015228
        """
        self.amin = amin
        self.aP = aP
        if aL is None:
            self.aL = 2.0*aP/3.0
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
        self.mnorm = scipy.integrate.quad(self.mnn, amin, self.aR,
                                          epsrel=1e-15, limit=100,
                                          points=[self.get_discontinuity()])[0]
        self.epstot = epstot

    def get_discontinuity(self):
        """ Find the  transition between bump aand powerlaw
        """
        # Need to solve fnn = b between aL and aP
        f = lambda x: self.fnn(x) - self.b(x)
        return scipy.optimize.bisect(f, self.aL, self.aP)

    def mnn(self, a):
        # Make sure we can handle both vector and scalar a
        a = np.asarray(a)
        scalar_input = False
        if a.ndim == 0:
            a = a[None]  # Makes a 1D
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
            return np.squeeze(ret).item()
        return np.reshape(ret, original_shape)

    def sigma0(self, a):
        ret = self.mnn(a)
        if self.epstot is not None:
            ret /= self.mnorm
            ret *= self.epstot
        return ret
