import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.integrate
import time
import hashlib
import pickle
import os.path
import multiprocessing
from abc import ABC, abstractmethod

# Modules from this package
from .taus_gridding import gridmap, get_gridding
from .power_bump import get_sigma0_birnstiel_bump, lognormpdf
from .power_bump import get_birnstiel_discontinuity
from .tanhsinh import TanhSinh


class StreamingSolver:
    q = 3.0/2.0
    Omega = 1.0
    kappa = 1.0
    rhog0 = 1.0
    r0 = 1.0

    def __init__(self, taus, epstot=0.1, beta=-3.5, sigdiff=None,
                 c=0.05, eta=0.05*0.05, alpha=None, dust_pressure=False):
        """
        Init for powerlaw PSI direct solver.

        Arguments:
        taus -- array of grid points in tau_s space
        epstot -- dut-to-gas mass density ratio
        beta -- powerlaw index of dust size-density, -3.5 is MRN
        cd -- dust fluid sound speed (default 0.0)
        sigdiff -- dust size diffusion coefficient (default 0.0)
        c -- gas sound speed (default 0.05)
        eta -- gas pressure gradient (default 0.05**2)
        alpha -- Turbulent alpha for gas momentum diffusion and
                 dust size diffusion.
        """
        self.taus = taus
        # this call is here so it is easy to override
        self.weights = self.get_trapz_weights()
        self.epstot = epstot
        self.beta = beta
        self.tsmax = taus.max()
        self.tsmin = taus.min()
        self.c = c
        self.eta = eta
        self.sigdiff = sigdiff  # optional size diffusion of dust fluid
        self.precalc_background_integrals()
        self.alpha = alpha  # alpha must be at least 1e-10
        self.dust_pressure = dust_pressure

    def precalc_background_integrals(self):
        """Precalculates integrals for the background state."""
        if not (self.tsmin < self.tsmax):
            raise ValueError('tsmin must be strictly smaller than tsmax')
        self.epsnorm = (self.tsmax**(4.0+self.beta))/(4.0+self.beta) \
                       - (self.tsmin**(4.0+self.beta))/(4.0+self.beta)
        self.deltatausmask = None
        # just do it by numerical quad, no special functions and cases
        # for SJP
        # self.J0 = scipy.integrate.quad(self.dJ0, self.tsmin, self.tsmax,
        #               limit=100, epsabs=1e-12, epsrel=1e-12)[0]
        # self.J1 = scipy.integrate.quad(self.dJ1, self.tsmin, self.tsmax,
        #               limit=100, epsabs=1e-12, epsrel=1e-12)[0]
        integrator = TanhSinh()
        self.J0 = integrator.integrate(self.dJ0, self.tsmin, self.tsmax)
        self.J1 = integrator.integrate(self.dJ1, self.tsmin, self.tsmax)

    def dJ0(self, taus):
        """Background state integrand for SJP """
        return 1.0/self.rhog0*self.sigma0(taus)/(1.0+self.kappa**2 * taus**2)

    def dJ1(self, taus):
        """Background state integrand for SJP """
        return 1.0/self.rhog0*self.sigma0(taus)*taus/(1.0+self.kappa**2
                                                      * taus**2)

    def sigma0(self, ts):
        """Powerlaw size density function,
           dust to gas as a function of stopping time
        """
        return np.where(np.logical_and((ts >= self.tsmin), (ts <= self.tsmax)),
                        self.epstot * ts**(3.0+self.beta) / self.epsnorm, 0.0)

    def background_velocities_SJP(self, taus):
        """Calculate the background velocities at dust sizes taus, use with SJP
           matrix.
        """
        J0 = self.J0
        J1 = self.J1
        vgx0 = 2.0*self.eta/(self.kappa) * J1/((1+J0)**2 + J1**2)
        vgy0 = - self.eta / self.Omega * (1.0+J0)/((1.0+J0)**2 + J1**2)
        # SJ u_x
        udx0 = 2.0*self.eta/(self.kappa) * (J1 - self.kappa*taus*(1+J0)) \
               / ((1 + self.kappa**2*taus**2)*((1+J0)**2 + J1**2))
        # SJ u_y
        udy0 = - self.eta / self.Omega * (1+J0+self.kappa*taus*J1) \
               / ((1 + self.kappa**2*taus**2)*((1+J0)**2 + J1**2))
        return [vgx0, vgy0, udx0, udy0]

    def get_trapz_weights(self):
        """Trapeziodal rule quadrature weights."""
        # use the non-uniform form for generality
        deltataus = self.taus[1:]-self.taus[:-1]
        trapzweight = np.zeros(self.taus.shape)
        trapzweight[:-1] = 0.5 * deltataus
        trapzweight[1:] += 0.5 * deltataus
        return trapzweight

    def build_system_matrix(self, Kx, Kz):
        """Build the linear system matrix for a specific wavevector Kx Kz."""
        self.build_system_matrix_SJP(Kx, Kz)

    def build_system_matrix_SJP(self, Kx, Kz):
        """
        System of equations in Sijme-Jan's notation and conventions
        """

        self.eigenvalues = None
        self.eigenvectors = None

        # undo nondim scaling of kx, kz
        self.kx = Kx/(self.eta*self.r0)
        self.kz = Kz/(self.eta*self.r0)
        # the right def for SJP notation? ex: 2010MNRAS.404L..64L
        self.S = self.q*self.Omega

        kx = self.kx
        kz = self.kz
        S = self.S
        c = self.c
        sigdiff = self.sigdiff
        rhog0 = self.rhog0
        Omega = self.kappa
        taus = self.taus
        self.ndust = len(taus)
        ndust = self.ndust
        self.nequations = 4+4*ndust
        nequations = self.nequations
        weight = self.weights
        sigma0 = self.sigma0(taus)
        vgx0, vgy0, udx0, udy0 = self.background_velocities_SJP(taus)

        A = np.zeros((nequations, nequations), dtype=np.complex)
        irhg = 0  # SJ rho_g^1
        ivgx = 1  # SJ v_{gx}^1
        ivgy = 2  # SJ v_{gy}^1
        ivgz = 3  # SJ v_{gz}^1
        # SJ single dust sigma = epsilon
        isig = [4 + 4*i for i in range(0, ndust)]
        iudx = [5 + 4*i for i in range(0, ndust)]
        iudy = [6 + 4*i for i in range(0, ndust)]
        iudz = [7 + 4*i for i in range(0, ndust)]
        self.irhg = irhg
        self.ivgx = ivgx
        self.ivgy = ivgy
        self.ivgz = ivgz
        self.isig = isig
        self.iudx = iudx
        self.iudy = iudy
        self.iudz = iudz

        # delta rho equation
        A[irhg, irhg] = kx*vgx0
        A[irhg, ivgx] = kx
        A[irhg, ivgz] = kz

        # vgx
        A[ivgx, irhg] = kx*c**2
        A[ivgx, ivgx] = kx*vgx0 - 1.0j/rhog0*(sigma0/taus*weight).sum()
        A[ivgx, ivgy] = 2.0j*Omega
        for idust, isign in enumerate(isig):
            A[ivgx, isign] = 1.0j/rhog0*(udx0[idust]-vgx0)/taus[idust] \
                            * weight[idust]
        for idust, iudxn in enumerate(iudx):
            A[ivgx, iudxn] = 1.0j/rhog0*sigma0[idust]/taus[idust] \
                            * weight[idust]

        # vgy
        A[ivgy, ivgx] = 1.0j*(S-2.0*Omega)
        A[ivgy, ivgy] = kx*vgx0 - 1.0j/rhog0*(sigma0/taus*weight).sum()
        for idust, isign in enumerate(isig):
            A[ivgy, isign] = 1.0j/rhog0*(udy0[idust]-vgy0)/taus[idust] \
                            * weight[idust]
        for idust, iudyn in enumerate(iudy):
            A[ivgy, iudyn] = 1.0j/rhog0*sigma0[idust]/taus[idust] \
                            * weight[idust]

        # vgz
        A[ivgz, irhg] = kz*c**2
        A[ivgz, ivgz] = kx*vgx0 - 1.0j/rhog0*(sigma0/taus*weight).sum()
        for idust, iudzn in enumerate(iudz):
            A[ivgz, iudzn] = 1.0j/rhog0*sigma0[idust]/taus[idust]*weight[idust]

        # sig
        for idust, isign in enumerate(isig):
            A[isign, isign] = kx*udx0[idust]
        for idust, (isign, iudxn) in enumerate(zip(isig, iudx)):
            A[isign, iudxn] = kx*sigma0[idust]
        for idust, (isign, iudzn) in enumerate(zip(isig, iudz)):
            A[isign, iudzn] = kz*sigma0[idust]

        # udx
        for idust, iudxn in enumerate(iudx):
            A[iudxn, irhg] = -1.0j/taus[idust]*(udx0[idust]-vgx0)
            A[iudxn, ivgx] = 1.0j/taus[idust]
            A[iudxn, iudxn] = kx*udx0[idust] - 1.0j/taus[idust]
        for (iudxn, iudyn) in zip(iudx, iudy):
            A[iudxn, iudyn] = 2.0j*Omega

        # udy
        for idust, iudyn in enumerate(iudy):
            A[iudyn, irhg] = -1.0j/taus[idust]*(udy0[idust]-vgy0)
            A[iudyn, ivgy] = 1.0j/taus[idust]
        for (iudxn, iudyn) in zip(iudx, iudy):
            A[iudyn, iudxn] = 1.0j*(S-2.0*Omega)
        for idust, iudyn in enumerate(iudy):
            A[iudyn, iudyn] = kx*udx0[idust] - 1.0j/taus[idust]

        # udz
        for idust, iudzn in enumerate(iudz):
            A[iudzn, ivgz] = 1.0j/taus[idust]
            A[iudzn, iudzn] = kx*udx0[idust] - 1.0j/taus[idust]

        self.linear_system_matrix = A
        if sigdiff is not None:
            self.add_size_diffusion()
        if self.alpha is not None:
            self.add_turbulence(Kx, Kz)
            if self.dust_pressure == True:
                self.add_dust_pressure()
        elif  self.dust_pressure == True:
            print("no dust pressure is possible with no alpha")

        # this line converts to PBL i.e. Benitez-Llambay eigenspace
        # self.linear_system_matrix = 1j*self.linear_system_matrix

    def add_dust_pressure(self):
        """Adds dust pressure terms to system matrix."""
        A = self.linear_system_matrix
        for idust, (isign, iudxn) in enumerate(zip(self.isig, self.iudx)):
            cdsquared = self.alpha*self.c**2/(1.0+self.taus[idust]**2)
            A[iudxn, isign] += self.kx*(cdsquared)  # dust pressure
        for idust, (isign, iudzn) in enumerate(zip(self.isig, self.iudz)):
            cdsquared = self.alpha*self.c**2/(1.0+self.taus[idust]**2)
            A[iudzn, isign] += self.kz*(cdsquared)  # dust pressure
        self.linear_system_matrix = A

    def add_size_diffusion(self):
        """ called by SJP form of system matrix
            Adds a diffusion operator in taus space on sigma, zero-flux at the
            endpoint boundaries, 2nd/3rd order stencils """
        A = self.linear_system_matrix
        # dust size disffusion
        for idust, isign in enumerate(self.isig):
            if idust == 0:
                # zero flux bc - zero first deriv at x0
                x0 = self.taus[0]
                xp1 = self.taus[1]
                xp2 = self.taus[2]
                # second order accurate
                # These code blocks break style guidelines, but they are
                # machine generated so that's best.
                # FortranForm[
                #  Collect[ExpandAll[
                #    D[D[cubicpoly /.
                #        Solve[{{cubicpoly /. s -> x0} ==
                #           f0, {D[cubicpoly, s] /. s -> x0} ==
                #           0, {cubicpoly /. s -> x1} == f1, {cubicpoly /. s -> x2 } ==
                #           f2}, {a, b, c, d}], s], s] /. s -> x0], {f0, f1, f2, f3},
                #   FullSimplify]]
                A[isign,isign]             +=  self.sigdiff *(-2*(3*x0**2 + xp1**2 + xp1*xp2 + xp2**2 - 3*x0*(xp1 + xp2)))/((x0 - xp1)**2*(x0 - xp2)**2)
                A[isign,self.isig[idust+1]]+=  self.sigdiff *(2*(x0 - xp2))/((x0 - xp1)**2*(xp1 - xp2))
                A[isign,self.isig[idust+2]]+=  self.sigdiff *(2*(x0 - xp1))/((x0 - xp2)**2*(-xp1 + xp2))
            if idust == 1:
                xm1 = self.taus[0]
                x0 = self.taus[1]
                xp1 = self.taus[2]
                # second order
                # FortranForm[
                #  Coefficient[
                #   Collect[ExpandAll[
                #     D[D[cubicpoly /.
                #         Solve[{{cubicpoly /. s -> xm1} ==
                #            fm1, {D[cubicpoly, s] /. s -> xm1} ==
                #            0, {cubicpoly /. s -> x0} ==
                #            f0, {cubicpoly /. s -> xp1 } == fp1}, {a, b, c, d}], s],
                #       s] /. s -> x0], {fm1, f0, fp1}, FullSimplify], fm1]]
                A[isign,self.isig[idust-1]]+= self.sigdiff * (-2/(x0 - xm1)**2 + 4/(xm1 - xp1)**2 + 4/((x0 - xm1)*(-xm1 + xp1)))
                A[isign,isign]             += self.sigdiff * ((6*x0 - 2*(2*xm1 + xp1))/((x0 - xm1)**2*(x0 - xp1)))
                A[isign,self.isig[idust+1]]+= self.sigdiff * ((-4*(x0 - xm1))/((x0 - xp1)*(xm1 - xp1)**2))
            elif idust == self.ndust-1:
                # zero flux bc - zero first deriv at x0
                x0 = self.taus[-1]
                xm1 = self.taus[-2]
                xm2 = self.taus[-3]
                # FortranForm[
                #  Collect[ExpandAll[
                #    D[D[cubicpoly /.
                #        Solve[{{cubicpoly /. s -> xm2} ==
                #           fm2, {D[cubicpoly, s] /. s -> x0} ==
                #           0, {cubicpoly /. s -> xm1} ==
                #           fm1, {cubicpoly /. s -> x0 } == f0}, {a, b, c, d}], s],
                #       s] /. s -> x0], {f0, fm1, fm2}, FullSimplify]]
                A[isign,self.isig[idust-2]]+=  self.sigdiff * (2*(x0 - xm1))/((x0 - xm2)**2*(-xm1 + xm2))
                A[isign,self.isig[idust-1]]+=  self.sigdiff * (2*(x0 - xm2))/((x0 - xm1)**2*(xm1 - xm2))
                A[isign,isign]             +=  self.sigdiff * (-2*(3*x0**2 + xm1**2 + xm1*xm2 + xm2**2 - 3*x0*(xm1 + xm2)))/((x0 - xm1)**2*(x0 - xm2)**2)
            elif idust == self.ndust-2:
                xm1 = self.taus[-3]
                x0 = self.taus[-2]
                xp1 = self.taus[-1]
                A[isign,self.isig[idust-1]]+=  self.sigdiff * (-2/(x0 - xm1)**2 + 4/(xm1 - xp1)**2 + 4/((x0 - xm1)*(-xm1 + xp1)))
                A[isign,isign]             +=  self.sigdiff * (6*x0 - 2*(2*xm1 + xp1))/((x0 - xm1)**2*(x0 - xp1))
                A[isign,self.isig[idust+1]]+=  self.sigdiff * (-4*(x0 - xm1))/((x0 - xp1)*(xm1 - xp1)**2)
            else:
                 xm2 = self.taus[idust-2]
                 xm1 = self.taus[idust-1]
                 x0 = self.taus[idust]
                 xp1 = self.taus[idust+1]
                 xp2 = self.taus[idust+2]
                 #third order, 5 point
                 #FortranForm[
                 # Coefficient[
                 #  Collect[ExpandAll[
                 #    D[D[InterpolatingPolynomial[{{xm2, fm2}, {xm1, fm1}, {x0,
                 #          f0}, {xp1, fp1}, {xp2, fp2}}, s], s], s] /. s -> x0], {f0,
                 #    fp1, fp2, fm1, fm2}, FullSimplify], fp2]]
                 A[isign,self.isig[idust-2]]+= self.sigdiff * (-2*(3*x0**2 + xp1*xp2 + xm1*(xp1 + xp2) - 2*x0*(xm1 + xp1 + xp2)))/((x0 - xm2)*(-xm1 + xm2)*(xm2 - xp1)*(xm2 - xp2))
                 A[isign,self.isig[idust-1]]+= self.sigdiff * (-2*(3*x0**2 + xp1*xp2 + xm2*(xp1 + xp2) - 2*x0*(xm2 + xp1 + xp2)))/((x0 - xm1)*(xm1 - xm2)*(xm1 - xp1)*(xm1 - xp2))
                 A[isign,isign]        += self.sigdiff * (2*(6*x0**2 + xm2*xp1 + (xm2 + xp1)*xp2 + xm1*(xm2 + xp1 + xp2) - 3*x0*(xm1 + xm2 + xp1 + xp2)))/((x0 - xm1)*(x0 - xm2)*(x0 - xp1)*(x0 - xp2))
                 A[isign,self.isig[idust+1]]+= self.sigdiff * (-2*(3*x0**2 + xm2*xp2 + xm1*(xm2 + xp2) - 2*x0*(xm1 + xm2 + xp2)))/((x0 - xp1)*(-xm1 + xp1)*(-xm2 + xp1)*(xp1 - xp2))
                 A[isign,self.isig[idust+2]]+= self.sigdiff * (-2*(3*x0**2 + xm2*xp1 + xm1*(xm2 + xp1) - 2*x0*(xm1 + xm2 + xp1)))/((x0 - xp2)*(-xm1 + xp2)*(-xm2 + xp2)*(-xp1 + xp2))
        self.linear_system_matrix = A

    def add_turbulence(self, Kx, Kz):
        """Add effects of turbulence to system matrix.
        Chen & Lin form, without the dust pressure of Umurham."""
        irho = self.irhg
        ivgx = self.ivgx
        ivgy = self.ivgy
        ivgz = self.ivgz

        A = self.linear_system_matrix

        kx = Kx/(self.eta*self.r0)
        kz = Kz/(self.eta*self.r0)
        ksq = kx*kx + kz*kz
        c = self.c
        alpha=self.alpha
        nu = alpha*c*c
        rhog0 = self.rhog0
        taus = self.taus
        sigma0 = self.sigma0(taus)
        vgx0, vgy0, udx0, udy0 = self.background_velocities_SJP(taus)
        ndust = len(taus)
        isig = [4 + 4*i for i in range(0, ndust)]
        # Dust diffusion
        # sigma
        for idust, isign in enumerate(isig):
            D = ((1+taus[idust]+4*taus[idust]**2)*alpha*c**2) \
                / ((1+taus[idust]**2)**2)
            A[isign, irho] += 1.0j*D*ksq*sigma0[idust]/rhog0
            A[isign, isign] -= 1.0j*D*ksq
        # Turbulence
        # vgx
        A[ivgx, ivgx] -= nu*((4/3)*kx**2 + kz**2)*1.0j
        A[ivgx, ivgz] -= (1/3)*nu*kx*kz*1.0j
        # vgy
        A[ivgy, ivgy] -= nu*(kz**2 + kx**2)*1.0j
        # vgz
        A[ivgz, ivgx] -= 1/3*nu*kx*kz*1.0j
        A[ivgz, ivgz] -= nu*(4/3*kz**2 + kx**2)*1.0j

        self.linear_system_matrix = A

    def del_system_matrix(self):
        del(self.linear_system_matrix)

    def solve_eigen(self):
        """Main compute routine, solve the eigenproblem and parse results."""
        self.eigenvalues, self.eigenvectors = \
            scipy.linalg.eig(self.linear_system_matrix)
        self.parse_fastest_dust_by_index()

    def solve_eigen_sparse(self, sigma, v0=None, tol=1e-10, k=1):
        """Find only eigenvalues close to guess sigma
        Not wildly useful in the end.
        """
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
                                        self.linear_system_matrix,
                                        k=k, sigma=sigma, v0=v0, tol=tol)
        if self.eigenvalues is not None:
            self.eigenvalues = np.concatenate((self.eigenvalues, eigenvalues))
            self.eigenvectors = np.concatenate(
                                     (self.eigenvectors, eigenvectors), axis=1)
        else:
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors

        self.parse_fastest_dust_by_index()

    def get_fastest_index(self):
        """Utility, the fastest growing eigenvalue is the most negative"""
        return np.argmax(self.eigenvalues.imag)

    def get_fastest_growth(self):
        return self.eigenvalues[self.get_fastest_index()]

    def get_fastest_err(self):
        """Compute an indication of how good the fastest growing
           eigenvector is"""
        reldiffs = np.matmul(self.linear_system_matrix,
                             self.eigenvectors[:, self.get_fastest_index()]) \
                   / self.eigenvectors[:, self.get_fastest_index()]
        maxerr = np.abs(reldiffs[np.argmax(np.abs(reldiffs))]
                        / self.get_fastest_growth() - 1.0)
        minerr = np.abs(reldiffs[np.argmin(np.abs(reldiffs))]
                        / self.get_fastest_growth() - 1.0)
        errfastest = np.max((maxerr, minerr))
        return errfastest

    def parse_fastest_dust_by_index(self):
        """Parse the fastest growing eigenmode to get the eigenfunctions in
           taus space"""
        self.fastest_eigenvec_by_taus, self.fastest_eigenvec_gas = \
                         self.parse_eigenvec_by_index(self.get_fastest_index())

    def parse_eigenvec_by_index(self, index):
        """Parse the fastest growing eigenmode to get the eigenfunctions in
           taus space"""
        # make the vgx perturbation amplitude 1.0 + 0.0j
        normb = lambda x: x/x[self.ivgx]
        # but then make the amplitude of max sigma 1.0
        norm = lambda x: normb(x)/np.abs(normb(x)[self.isig]).max()
        eigvec = norm(self.eigenvectors[:, index])
        by_index = np.zeros((len(self.taus)), dtype=[('sig', np.complex),
                                                     ('udx', np.complex),
                                                     ('udy', np.complex),
                                                     ('udz', np.complex)])
        for i in range(0, len(self.taus)):
            by_index['sig'][i] = eigvec[self.isig[i]]
            by_index['udx'][i] = eigvec[self.iudx[i]]
            by_index['udy'][i] = eigvec[self.iudy[i]]
            by_index['udz'][i] = eigvec[self.iudz[i]]
        return (by_index,
                {'rho': eigvec[self.irhg], 'vgx': eigvec[self.ivgx],
                 'vgy': eigvec[self.ivgy], 'vgz': eigvec[self.ivgz]})


# Perhaps this can get deleted
class StreamingSolverGaussLegendre(StreamingSolver):
    """Power law dust distribution, but with Gauss-Legengdre quadrature.
    """

    def __init__(self, tsint, ntaus, epstot=0.1, beta=-3.5,
                 c=0.05, eta=0.05*0.05, alpha=None, dust_pressure=False):
        self.epstot = epstot
        self.beta = beta
        self.tsmax = tsint.max()
        self.tsmin = tsint.min()
        self.taus, self.weights = self.get_legendre_weights(ntaus)
        self.precalc_background_integrals()
        self.c = c
        self.eta = eta
        self.alpha = alpha
        self.dust_pressure = dust_pressure

    def get_legendre_weights(self, ndust):
        x, w = scipy.special.roots_legendre(ndust)
        taus = (x+1.0)*0.5*(self.tsmax-self.tsmin) + self.tsmin
        legendreweight = w*(self.tsmax-self.tsmin)/2.0
        return taus, legendreweight


class StreamingSolverLogNormal(StreamingSolver):
    """With a truncated LogNormal dust distribution
       sigma is the width, and peak is the peak size.
    """

    def __init__(self, taus, epstot, lognormsigma, peak, sigdiff=None,
                 c=0.05, eta=0.05*0.05, alpha=None, dust_pressure=False):
        self.taus = taus
        self.epstot = epstot
        self.lognormsigma = lognormsigma
        self.peak = peak
        self.eta = eta
        self.c = c
        self.sigdiff = sigdiff
        self.alpha = alpha
        self.dust_pressure = dust_pressure
        self.tsmax = taus.max()
        self.tsmin = taus.min()
        self.weights = self.get_trapz_weights()
        self.deltatausmask = None
        # self.lognormpdf = \
        #     scipy.stats.lognorm(s=self.sigma, loc=-1.0+peak).pdf
        self.lognormpdf = lambda x: lognormpdf(x, s=self.lognormsigma,
                                               loc=-1.0+peak)
        self.massdist = lambda a: self.lognormpdf(a) * a**3
        self.epsnorm = scipy.integrate.quad(self.massdist,
                                            self.tsmin, self.tsmax,
                                            limit=100, epsabs=1e-12,
                                            epsrel=1e-12)[0]
        # just do it by numerical quad, no special functions and cases
        # for SJP
        # self.J0 = scipy.integrate.quad(self.dJ0, self.tsmin, self.tsmax,
        #                                limit=100, epsabs=1e-12,
        #                                epsrel=1e-12)[0]
        # self.J1 = scipy.integrate.quad(self.dJ1, self.tsmin, self.tsmax,
        #                                limit=100, epsabs=1e-12,
        #                                epsrel=1e-12)[0]
        integrator = TanhSinh()
        self.J0 = integrator.integrate(self.dJ0, self.tsmin, self.tsmax)
        self.J1 = integrator.integrate(self.dJ1, self.tsmin, self.tsmax)

    def sigma0(self, ts):
        return self.massdist(ts)/self.epsnorm * self.epstot


class StreamingSolverPowerBump(StreamingSolver):
    """With the cratering-bump inspired bump dust distribution
       sigma is the width, and peak is the peak size"""

    def __init__(self, taus, epstot, beta, aL, aP, bumpfac,
                 sigdiff=None, c=0.05, eta=0.05*0.05, alpha=None, dust_pressure=False):
        self.taus = taus
        self.epstot = epstot
        self.beta = beta
        self.eta = eta
        self.c = c
        self.alpha = alpha
        self.dust_pressure = dust_pressure
        self.tsmax = taus.max()
        self.tsmin = taus.min()
        self.weights = self.get_trapz_weights()
        self.aL = aL
        self.aP = aP
        self.bumpfac = bumpfac
        self.sigdiff = sigdiff
        self.deltatausmask = None
        self.pbdisc = get_birnstiel_discontinuity(
                                                 amin=self.tsmin,
                                                 aL=self.aL,
                                                 aP=self.aP,
                                                 aR=self.tsmax,
                                                 bumpfac=self.bumpfac,
                                                 beta=self.beta)
        self.massdist = np.vectorize(get_sigma0_birnstiel_bump(amin=self.tsmin,
                                                 aL=self.aL,
                                                 aP=self.aP,
                                                 aR=self.tsmax,
                                                 bumpfac=self.bumpfac,
                                                 beta=self.beta,
                                                 epstot=self.epstot))
        # Quadpack isn't the best, but we do what we can
        epsrel = 1e-10
        epsabs = 1e-10
        integrator = TanhSinh()
        self.epsnorm = scipy.integrate.quad(self.massdist,
                                            self.tsmin, self.tsmax, limit=100,
                                            epsabs=epsabs, epsrel=epsrel,
                                            points=[self.pbdisc])[0]
        self.J0 = integrator.integrate(self.dJ0, self.tsmin, self.tsmax)
        #self.J0 = scipy.integrate.quad(self.dJ0, self.tsmin, self.tsmax,
        #                               limit=100, epsabs=epsabs,
        #                               epsrel=epsrel,
        #                               points=[self.pbdisc])[0]
        self.J1 = integrator.integrate(self.dJ1, self.tsmin, self.tsmax)
        #self.J1 = scipy.integrate.quad(self.dJ1, self.tsmin, self.tsmax,
        #                               limit=100, epsabs=epsabs,
        #                               epsrel=epsrel,
        #                               points=[self.pbdisc])[0]

    def sigma0(self, ts):
        return self.massdist(ts)


class ConvergerBase(ABC):
    """Object for running a convergence study on a single modee"""

    picklenameprefix = 'converger_'

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_StreamingSolver(self, ntaus):
        pass

    def runcompute(self):
        fastesteigens = np.zeros(self.refine, dtype=np.complex)
        alleigens = []
        computetimes = np.zeros(self.refine)
        ss = self.get_StreamingSolver(2**self.ll+1)
        ss.build_system_matrix(self.Kx, self.Kz)
        ss.solve_eigen()
        ss.del_system_matrix()
        # fastestgrowing = ss.get_fastest_growth()
        for ill, ls in enumerate(range(self.ll+1,
                                 self.ll+fastesteigens.shape[0]+1)):
            start = time.perf_counter()
            # split compute off into a separate process to make sure memory
            # gets freed up
            sharedeigensreal = multiprocessing.RawArray('d', (2**ls+1)*4+4)
            sharedeigensimag = multiprocessing.RawArray('d', (2**ls+1)*4+4)
            p = multiprocessing.Process(target=self.isolatedcompute,
                                        args=(ls, sharedeigensreal,
                                              sharedeigensimag))
            p.start()
            p.join()
            theseeigens = np.zeros((2**ls+1)*4+4, dtype=np.complex)
            theseeigens.real = sharedeigensreal
            theseeigens.imag = sharedeigensimag
            alleigens.append(theseeigens)
            # this could be a trip-up, this is a seperate place outside of
            # StreamingSolver where the fastest growing mode is chosen
            fastesteigens[ill] = theseeigens[np.argmax(theseeigens.imag)]
            del(sharedeigensreal)
            del(sharedeigensimag)
            end = time.perf_counter()
            computetimes[ill] = end-start
        self.fastesteigens = fastesteigens
        self.alleigens = alleigens
        self.computetimes = computetimes
        self.write_state()

    def isolatedcompute(self, ls, sharedeigensreal, sharedeigensimag):
        """Called in a multiprocessing process,
            passes data back with shared arrays """
        # Used to have a queue, seemd to hang once in a while on the join
        ss = self.get_StreamingSolver(2**ls+1)
        ss.build_system_matrix(self.Kx, self.Kz)
        ss.solve_eigen()
        ss.del_system_matrix()
        # Avoid the queue and pack it up for the shared arrays backi
        # in the main process
        # fastestgrowing = ss.get_fastest_growth()
        # queue.put(fastestgrowing)
        realpart = np.frombuffer(sharedeigensreal, dtype=np.float64)
        np.copyto(realpart, ss.eigenvalues.real)
        imagpart = np.frombuffer(sharedeigensimag, dtype=np.float64)
        np.copyto(imagpart, ss.eigenvalues.imag)

    def write_state(self):
        print('Writing ', self.picklename)
        with open(self.picklename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def backtraceeigen(self, value, startlevel):
        """ Trace the convergence of an eigenvalue, matches up the chain of
            closest eigenvlues in the series of computations.

            trace, tracevals = backtraceeigen(self, value, startlevel)

            value : starting point for search in the complex plane
            startlevel: level in self.alleigens list to start tracing from
            trace: list of indicies of matched eigenvalues
            tracevals: list of eigenvalues
        """
        nlevels = len(self.alleigens)
        trace = np.zeros(nlevels, dtype=np.int)
        trace[startlevel] = np.argmin(np.abs(self.alleigens[startlevel]
                                             - value))
        if startlevel < nlevels-1:
            # run up to top
            for level in range(startlevel+1, nlevels):
                trace[level] = np.argmin(np.abs(self.alleigens[level]
                                    - self.alleigens[level-1][trace[level-1]]))
        if startlevel > 0:
            # down to bottom
            for level in range(startlevel-1, -1, -1):
                trace[level] = np.argmin(np.abs(self.alleigens[level]
                                    - self.alleigens[level+1][trace[level+1]]))
        tracevals = np.array([self.alleigens[il][trace[il]]
                              for il in range(nlevels)])
        return trace, tracevals



class Converger(ConvergerBase):
    """Convergence study with a power-law dust distribution"""

    def __init__(self, tsint, epstot, beta, Kx, Kz, ll=6, refine=5,
                 prefix='outputs', gridding='chebyshevroots', alpha=None,
                 dust_pressure=False):
        self.tsint = tsint
        self.epstot = epstot
        self.beta = beta
        self.Kx = Kx
        self.Kz = Kz
        self.ll = ll
        self.refine = refine
        self.reltol = 1e-14
        self.alpha = alpha
        self.dust_pressure = dust_pressure
        if gridding not in gridmap.keys():
            raise ValueError('Unknown gridding ', gridding)
        self.gridding = gridmap[gridding]
        repstr = ''
        # this is for backwards compatibility, peel off old limited cases
        if epstot == 1.0 and gridding == 'linear':
            for val in [tsint[0], tsint[1], beta, Kx, Kz, ll, refine, alpha]:
                repstr += '{:e}'.format(val)
        else:
            for val in [epstot, tsint[0], tsint[1], beta, Kx, Kz, ll, refine]:
                repstr += '{:e}'.format(val)
            repstr += gridding
        if alpha is not None:
            repstr += '{:e}'.format(alpha)
        if dust_pressure:
            repstr += 'dust_pressure'
        self.picklename = os.path.join(prefix, self.picklenameprefix
                              + hashlib.md5(
                                  repstr.encode(encoding='utf-8')).hexdigest()
                              + '.pickle')

    def get_StreamingSolver(self, ntaus):
        taus = get_gridding(self.gridding, self.tsint, ntaus)
        return StreamingSolver(taus, epstot=self.epstot, beta=self.beta,
                               alpha=self.alpha,
                               dust_pressure=self.dust_pressure)


class ConvergerLogNormal(ConvergerBase):
    """Convergence study with a LogNormal dust distribution"""
    def __init__(self, tsint, epstot, sigma, peak, Kx, Kz, ll=6,
                 refine=5, prefix='outputs', gridding='chebyshevroots',
                 alpha=None, dust_pressure=False):
        self.tsint = tsint
        self.epstot = epstot
        self.sigma = sigma
        self.peak = peak
        self.Kx = Kx
        self.Kz = Kz
        self.ll = ll
        self.refine = refine
        self.reltol = 1e-14
        self.alpha = alpha
        self.dust_pressure = dust_pressure
        if gridding not in gridmap.keys():
            raise ValueError('Unknown gridding ', gridding)
        self.gridding = gridmap[gridding]
        repstr = 'ConvergerLogNormal'
        for val in [epstot, tsint[0], tsint[1], sigma, peak, Kx, Kz,
                    ll, refine]:
            repstr += '{:e}'.format(val)
        if alpha is not None:
            repstr += '{:e}'.format(alpha)
        if dust_pressure:
            repstr += 'dust_pressure'
        repstr += gridding
        self.picklename = \
            os.path.join(prefix,
                         self.picklenameprefix
                         + hashlib.md5(
                             repstr.encode(encoding='utf-8')).hexdigest()
                         + '.pickle')

    def get_StreamingSolver(self, ntaus):
        return StreamingSolverLogNormal(
                              np.linspace(self.tsint[0], self.tsint[1], ntaus),
                              epstot=self.epstot,
                              lognormsigma=self.sigma,
                              peak=self.peak,
                              alpha=self.alpha,
                              dust_pressure=self.dust_pressure)


class ConvergerPowerBump(Converger):
    """Convergence study with a power-law + lognormal bump dust distribution"""

    def __init__(self, tsint, epstot, beta, aL, aP, bumpfac, Kx, Kz, ll=6,
                 refine=5, prefix='outputs', gridding='chebyshevroots',
                 alpha=None, dust_pressure=False):
        self.tsint = tsint
        self.epstot = epstot
        self.beta = beta
        self.aL = aL
        self.aP = aP
        self.bumpfac = bumpfac
        self.alpha = alpha
        self.dust_pressure = dust_pressure
        self.Kx = Kx
        self.Kz = Kz
        self.ll = ll
        self.refine = refine
        self.reltol = 1e-14
        if gridding not in gridmap.keys():
            raise ValueError('Unknown gridding ', gridding)
        self.gridding = gridmap[gridding]
        repstr = 'ConvergerPowerBump'
        for val in [epstot, tsint[0], tsint[1], beta, aL, aP, bumpfac,
                    Kx, Kz, ll, refine]:
            repstr += '{:e}'.format(val)
        if alpha is not None:
            repstr += '{:e}'.format(alpha)
        if dust_pressure:
            repstr += 'dust_pressure'
        repstr += gridding
        self.picklename = os.path.join(prefix,
                              self.picklenameprefix
                              + hashlib.md5(
                                  repstr.encode(encoding='utf-8')).hexdigest()
                              + '.pickle')

    def get_StreamingSolver(self, ntaus):
        taus = get_gridding(self.gridding, self.tsint, ntaus)
        return StreamingSolverPowerBump(taus, epstot=self.epstot,
                                        beta=self.beta, aL=self.aL,
                                        aP=self.aP, bumpfac=self.bumpfac,
                                        alpha=self.alpha,
                                        dust_pressure=self.dust_pressure)
