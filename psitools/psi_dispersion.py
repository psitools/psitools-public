#!/usr/bin/python
#
# Copyright 2020 Colin McNally, Sijme-Jan Paardekooper, Francesco Lovascio
#    colin@colinmcnally.ca, s.j.paardekooper@qmul.ac.uk, f.lovascio@qmul.ac.uk
#
#    This file is part of psitools.
#
#    psitools is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    psitools is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with psitools.  If not, see <https://www.gnu.org/licenses/>.
#
# NB: Sijme-Jan wrote this.
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import warnings

from .sizedensity import SizeDensity


class PSIDispersion():
    """Class holding the full PSI dispersion relation.

    If the PSI dispersion relation is given by f(w) = 0, calculate(w) gives the value of f at w.

    Args:
        dust_to_gas_ratio: Dust to gas ratio
        stokes_range: Range of Stokes numbers to consider
        sound_speed_over_eta (optional) : Gas sound speed divided by eta. Defaults to 1/0.05.
        size_distribution_power (optional): Power law index of size distribution. Defaults to 3.5 = MRN.
        single_size_flag (optional): If true, calculate for single size at maximum Stokes number.
        size_density (optional): Instance of SizeDensity class. Defaults to None, in which case a power law size density is constructed from size_distribution_power.
    """
    def __init__(self,
                 dust_to_gas_ratio,
                 stokes_range,
                 sound_speed_over_eta=1/0.05,
                 size_distribution_power=3.5,
                 single_size_flag=False,
                 size_density=None,
                 tanhsinh_integrator=None):
        self.mu = dust_to_gas_ratio
        self.taumin = stokes_range[0]
        self.taumax = stokes_range[1]
        self.c = sound_speed_over_eta
        self.ssf = single_size_flag

        self.size_density = size_density
        if self.size_density is None:
            # Create power law SizeDensity
            sigma = lambda x: np.power(x, 3 - size_distribution_power)
            sigma_integral = (stokes_range[0]**(4.0-size_distribution_power)
                              -stokes_range[1]**(4.0-size_distribution_power))\
                             / (size_distribution_power - 4.0)
            self.size_density = SizeDensity(sigma, stokes_range,
                                            sigma_integral=sigma_integral)

        self.poles = list(self.size_density.poles)
        self.tanhsinh_integrator = tanhsinh_integrator

        # Error tolerances in calls to quadpack
        self.quad_epsrel = 1.49e-12
        self.quad_epsabs = 1.49e-9

        # Equilibrium dust and gas velocities
        j0 = np.real(self.int_j(0))
        j1 = np.real(self.int_j(1))
        denom = 1/((1 + j0)**2 + j1**2)
        self.vgx = 2*j1*denom
        self.vgy = -(1 + j0)*denom
        self.ux = lambda x: 2*denom*(j1 - x*(1 + j0))/(1 + x*x)
        self.uy = lambda x: -denom*(1 + j0 + x*j1)/(1 + x*x)

        # Parts of coefficients of polynomial Re(w) - Kx*ux(x) = 0
        self.base_poly = [-2*denom*j1, 2*denom*(1 + j0), 0]

    def correct_integral(self, f, a, b, poles):
        """Separately integrate troublesome subintervals"""

        # The integral that gives trouble; full_output=1 prevents a warning
        res, err, info, msg = \
          integrate.quad(f, a, b, points=poles, full_output=1,
                         epsabs=self.quad_epsabs,
                         epsrel=self.quad_epsrel)

        # Original result summed over subintervals
        res = np.sum(info['rlist'][0:info['last']])

        # Select subintervals that have an error that is too large
        sel = np.asarray(info['elist'][0:info['last']] >
                         self.quad_epsabs).nonzero()

        # Loop through all troublesome subintervals
        for i in sel[0]:
            # Subinterval range and result
            a_max = info['alist'][i]
            b_max = info['blist'][i]
            r_max = info['rlist'][i]

            # Integrate subinterval separately
            try:
                r_sub = integrate.quad(f, a_max, b_max,
                                       epsabs=self.quad_epsabs,
                                       epsrel=self.quad_epsrel)[0]
            except:
                # Repeat recursively if necessary
                r_sub = self.correct_integral(f, a_max, b_max, [])

            # Update result with new value for subinterval
            res = res - r_max + r_sub

        # Return new result
        return res

    def average(self, g):
        """Calculate the integral (1/rho_g) \int sigma * g(tau) da.

        The function g should be a function of stopping time only."""

        # Single size at tau_max
        if (self.ssf == True):
            return self.mu*g(self.taumax)

        # Deal with power law size distribution.
        # Set s = a/amax = tau/taumax and integrate over s.
        t = self.taumax
        s_min = self.taumin/self.taumax

        # Sigma(s) = amax*sigma(amax*s)/rho_d
        Sigma = lambda x: self.size_density(x)
        a = np.log(s_min)
        b = 0
        poles = np.log(self.poles)

        if self.tanhsinh_integrator is not None:
            f = lambda u: g(t*np.exp(u))*np.exp(u)*Sigma(np.exp(u))

            if len(poles) == 0:
                ret = self.tanhsinh_integrator.integrate(f, a, b)
            else:
                a_list = np.concatenate((np.asarray([a]),poles))
                b_list = np.concatenate((poles, np.asarray([b])))

                ret = 0.0
                for aa, bb in zip(a_list, b_list):
                    ret += self.tanhsinh_integrator.integrate(f, aa, bb)

            return self.mu*ret

        # Not using tanh-sinh, use quadpack
        fR = lambda u: np.real(g(t*np.exp(u)))*np.exp(u)*Sigma(np.exp(u))
        fI = lambda u: np.imag(g(t*np.exp(u)))*np.exp(u)*Sigma(np.exp(u))

        # Make warnings errors that we can catch
        warnings.simplefilter('error', integrate.IntegrationWarning)
        try:
            retR = integrate.quad(fR, a, b, points=poles,
                                  epsabs=self.quad_epsabs,
                                  epsrel=self.quad_epsrel)[0]
        except:
            retR = self.correct_integral(fR, a, b, poles)
        try:
            retI = integrate.quad(fI, a, b, points=poles,
                                  epsabs=self.quad_epsabs,
                                  epsrel=self.quad_epsrel)[0]
        except:
            retI = self.correct_integral(fI, a, b, poles)
        # Back to default warning behaviour
        warnings.simplefilter('default', integrate.IntegrationWarning)

        # Return result as complex
        return self.mu*(retR + 1j*retI)

    def int_j(self, alpha):
        """Calculate the J_alpha integral, needed for the background velocities"""

        g = lambda x: np.power(x, alpha)/(1 + x**2)
        return self.average(g)

    def matrix_P(self, w):
        """Matrix P: linear gas momentum equation, excluding drag terms,
            is P*v_g = 0"""
        d = self.kx*self.vgx - w - 1j*self.nu*(self.kx**2 + self.kz**2)

        Pxx = d + (self.kx*self.c)**2/(w - self.kx*self.vgx) - \
          1j*self.kx**2*self.nu/3
        Pxy = 2*1j
        Pxz = self.kx*(self.c)**2*self.kz/(w - self.kx*self.vgx) - \
          1j*self.kx*self.kz*self.nu/3

        Pyx = -0.5*1j
        Pyy = d
        Pyz = 0

        Pzx = self.kz*(self.c)**2*self.kx/(w - self.kx*self.vgx) - \
          1j*self.kx*self.kz*self.nu/3
        Pzy = 0
        Pzz = d + (self.kz*self.c)**2/(w - self.kx*self.vgx) - \
          1j*self.kz**2*self.nu/3

        return np.asarray([[Pxx, Pxy, Pxz], [Pyx, Pyy, Pyz], [Pzx, Pzy, Pzz]])

    def matrix_inverse_A(self, w):
        """Calculate the inverse of matrix A. Linear dust momentum equation is A*u = D*v_g."""
        d = lambda x: self.kx*self.ux(x) - w - 1j/x
        denom = lambda x: 1/(1 - d(x)**2)

        a11 = lambda x:-d(x)*denom(x)
        a12 = lambda x:2*1j*denom(x)
        a13 = lambda x:0*x
        a21 = lambda x:-0.5*1j*denom(x)
        a22 = lambda x:-d(x)*denom(x)
        a23 = lambda x:0*x
        a31 = lambda x:0*x
        a32 = lambda x:0*x
        a33 = lambda x:1/d(x)

        return [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]

    def matrix_AD(self, A, w):
        """Matrix AD: Linear dust momentum equation is A*u = D*v_g.
            Returns A^(-1)*D."""
        # Unperturbed relative velocities
        dux = lambda x: self.ux(x) - self.vgx
        duy = lambda x: self.uy(x) - self.vgy

        denom = 1/(w - self.kx*self.vgx)

        # Elements of matrix D
        d11 = lambda x: (1j/x)*(dux(x)*self.kx*denom - 1)
        d12 = lambda x: 0*x
        d13 = lambda x: (1j/x)*dux(x)*self.kz*denom
        d21 = lambda x: (1j/x)*duy(x)*self.kx*denom
        d22 = lambda x: -1j/x
        d23 = lambda x: (1j/x)*duy(x)*self.kz*denom
        d31 = lambda x: 0*x
        d32 = lambda x: 0*x
        d33 = lambda x: -1j/x

        # Matrix multiply A and D
        ad11 = lambda x: (A[0][0](x)*d11(x) +
                          A[0][1](x)*d21(x) +
                          A[0][2](x)*d31(x))
        ad12 = lambda x: (A[0][0](x)*d12(x) +
                          A[0][1](x)*d22(x) +
                          A[0][2](x)*d32(x))
        ad13 = lambda x: (A[0][0](x)*d13(x) +
                          A[0][1](x)*d23(x) +
                          A[0][2](x)*d33(x))
        ad21 = lambda x: (A[1][0](x)*d11(x) +
                          A[1][1](x)*d21(x) +
                          A[1][2](x)*d31(x))
        ad22 = lambda x: (A[1][0](x)*d12(x) +
                          A[1][1](x)*d22(x) +
                          A[1][2](x)*d32(x))
        ad23 = lambda x: (A[1][0](x)*d13(x) +
                          A[1][1](x)*d23(x) +
                          A[1][2](x)*d33(x))
        ad31 = lambda x: (A[2][0](x)*d11(x) +
                          A[2][1](x)*d21(x) +
                          A[2][2](x)*d31(x))
        ad32 = lambda x: (A[2][0](x)*d12(x) +
                          A[2][1](x)*d22(x) +
                          A[2][2](x)*d32(x))
        ad33 = lambda x: (A[2][0](x)*d13(x) +
                          A[2][1](x)*d23(x) +
                          A[2][2](x)*d33(x))

        return [[ad11, ad12, ad13], [ad21, ad22, ad23], [ad31, ad32, ad33]]

    def matrix_VAD(self, AD, w):
        """Matrix VAD: V is defined so that \Delta u^0 \sigma/\sigma^0 = (V - I)u. Returns product V*A^(-1)*D."""

        # Unperturbed relative velocities
        dux = lambda x: self.ux(x) - self.vgx
        duy = lambda x: self.uy(x) - self.vgy

        denom = lambda x:1/(w - self.kx*self.ux(x) + \
                            1j*((self.kx**2 + self.kz**2)*self.D(x) + 1.0e-10))

        # Elements of matrix V
        v11 = lambda x: 1 + dux(x)*self.kx*denom(x)
        v12 = lambda x: 0*x
        v13 = lambda x: dux(x)*self.kz*denom(x)
        v21 = lambda x: duy(x)*self.kx*denom(x)
        v22 = lambda x: 1
        v23 = lambda x: duy(x)*self.kz*denom(x)
        v31 = lambda x: 0*x
        v32 = lambda x: 0*x
        v33 = lambda x: 1

        # Matrix multiply V and AD
        vd11 = lambda x: (v11(x)*AD[0][0](x) +
                          v12(x)*AD[1][0](x) +
                          v13(x)*AD[2][0](x))
        vd12 = lambda x: (v11(x)*AD[0][1](x) +
                          v12(x)*AD[1][1](x) +
                          v13(x)*AD[2][1](x))
        vd13 = lambda x: (v11(x)*AD[0][2](x) +
                          v12(x)*AD[1][2](x) +
                          v13(x)*AD[2][2](x))
        vd21 = lambda x: (v21(x)*AD[0][0](x) +
                          v22(x)*AD[1][0](x) +
                          v23(x)*AD[2][0](x))
        vd22 = lambda x: (v21(x)*AD[0][1](x) +
                          v22(x)*AD[1][1](x) +
                          v23(x)*AD[2][1](x))
        vd23 = lambda x: (v21(x)*AD[0][2](x) +
                          v22(x)*AD[1][2](x) +
                          v23(x)*AD[2][2](x))
        vd31 = lambda x: (v31(x)*AD[0][0](x) +
                          v32(x)*AD[1][0](x) +
                          v33(x)*AD[2][0](x))
        vd32 = lambda x: (v31(x)*AD[0][1](x) +
                          v32(x)*AD[1][1](x) +
                          v33(x)*AD[2][1](x))
        vd33 = lambda x: (v31(x)*AD[0][2](x) +
                          v32(x)*AD[1][2](x) +
                          v33(x)*AD[2][2](x))

        return [[vd11, vd12, vd13], [vd21, vd22, vd23], [vd31, vd32, vd33]]

    def matrix_W(self, w):
        # Unperturbed relative velocities
        dux = lambda x: self.ux(x) - self.vgx
        duy = lambda x: self.uy(x) - self.vgy

        k2 = self.kx**2 + self.kz**2
        denom = lambda x:1j*self.D(x)*k2/(w - self.kx*self.ux(x) + \
                            1j*(k2*self.D(x) + 1.0e-10))/(w - self.kx*self.vgx)

        # Elements of matrix W
        w11 = lambda x: dux(x)*self.kx*denom(x) - 1
        w12 = lambda x: 0*x
        w13 = lambda x: dux(x)*self.kz*denom(x)
        w21 = lambda x: duy(x)*self.kx*denom(x)
        w22 = lambda x: -1
        w23 = lambda x: duy(x)*self.kz*denom(x)
        w31 = lambda x: 0*x
        w32 = lambda x: 0*x
        w33 = lambda x: -1

        return [[w11, w12, w13], [w21, w22, w23], [w31, w32, w33]]

    def matrix_M(self, w):
        """Matrix M: Drag terms in linear gas momentum equation are M*v_g."""
        A = self.matrix_inverse_A(w)
        AD = self.matrix_AD(A, w)
        VAD = self.matrix_VAD(AD, w)
        W = self.matrix_W(w)

        K = lambda x: 1j/x

        m11 = self.average(lambda x: (VAD[0][0](x) + W[0][0](x))*K(x))
        m12 = self.average(lambda x: (VAD[0][1](x) + W[0][1](x))*K(x))
        m13 = self.average(lambda x: (VAD[0][2](x) + W[0][2](x))*K(x))
        m21 = self.average(lambda x: (VAD[1][0](x) + W[1][0](x))*K(x))
        m22 = self.average(lambda x: (VAD[1][1](x) + W[1][1](x))*K(x))
        m23 = self.average(lambda x: (VAD[1][2](x) + W[1][2](x))*K(x))
        #m31 = self.average(lambda x: (VAD[2][0](x) + W[2][0](x))*K(x))
        #m32 = self.average(lambda x: (VAD[2][1](x) + W[2][1](x))*K(x))
        m33 = self.average(lambda x: (VAD[2][2](x) + W[2][2](x))*K(x))

        return np.asarray([[m11, m12, m13], [m21, m22, m23], [0, 0, m33]])

    def calculate(self, w, wave_number_x, wave_number_z, viscous_alpha=0):
        """If the dispersion relation is f(w) = 0, this function calculates f at w.

        Args:
            w: Complex frequency at which to evaluate the dispersion relation
            wave_number x: Dimensionless wave number (YG05) in x
            wave_number_z: Dimensionless wave number (YG05) in z
            viscous_alpha (optional): Background turbulent gas viscosity parameter. Defaults to zero.
        """
        # Lazy; avoids passing wave numbers to all functions
        self.kx = wave_number_x
        self.kz = wave_number_z

        # Viscosity and dust diffusion
        self.nu = viscous_alpha*self.c**2
        self.D = lambda x: (1 + x + 4*x*x)*self.nu/(1 + x*x)**2

        # Make sure we can handle both vector and scalar w
        w = np.asarray(w)
        scalar_input = False
        if w.ndim == 0:
            w = w[None]  # Makes w 1D
            scalar_input = True
        else:
            original_shape = np.shape(w)
            w = np.ravel(w)

        # Calculate dispersion relation
        ret = 0.0*w
        for i in range(0, len(w)):
            # Identify the pole (can there be 2???)
            #f = lambda x: np.real(w[i]) - self.kx*self.ux(x)

            # List of poles from size density; may be empty
            # Scale to taumax
            self.poles = list(np.array(self.size_density.poles)/self.taumax)

            # Polynomial solving Re(w) - Kx*ux(x) = 0
            coeff = [self.base_poly[0] + np.real(w[i])/self.kx,
                     self.base_poly[1], np.real(w[i])/self.kx]
            roots = np.polynomial.Polynomial(coeff).roots()

            # Select roots in domain
            for j in range(0, len(roots)):
                if (np.real(roots[j]) > self.taumin and
                    np.real(roots[j]) < self.taumax and
                    np.isreal(roots[j])):
                     self.poles.append(roots[j]/self.taumax)

            ## if (f(self.taumin)*f(self.taumax) < 0.0):
            ##     res = opt.fsolve(f, x0=1.5*self.taumin + 0.5*self.taumax,
            ##                      xtol=1.49e-11)
            ##     if res[0] > self.taumin:
            ##         self.poles.append(res[0]/self.taumax)
            ## else:
            ##     # Possibly two poles
            ##     res = opt.fsolve(f, x0=self.taumax, xtol=1.49e-11)
            ##     if (res[0] > self.taumin and res[0] < self.taumax):
            ##          self.poles.append(res[0]/self.taumax)
            ##     res = opt.fsolve(f, x0=self.taumin, xtol=1.49e-11)
            ##     if (res[0] > self.taumin and res[0] < self.taumax):
            ##          self.poles.append(res[0]/self.taumax)

            ## print('Old: ', self.poles)
            #for p in self.poles:
            #    p = p/self.taumax
            #for j in range(0, len(self.poles)):
            #    self.poles[j] /= self.taumax
            #if len(self.poles) > 0:
            #    self.poles = self.poles.sort()

            self.poles.sort()

            M = self.matrix_M(w[i])
            P = self.matrix_P(w[i])

            ret[i] = np.linalg.det(P + M)

        # Return value of original shape
        if scalar_input:
            return np.squeeze(ret)
        return np.reshape(ret, original_shape)

    def eigenvector(self, w, wave_number_x, wave_number_z, viscous_alpha=0):
        """Calculate the eigenvector for eigenvalue w.

        Args:
            w: Eigenvalue as previously found from self.calculate().
            wave_number_x: Nondimensional (using YG05) x wave number
            wave_number_z: Nondimensional (using YG05) z wave number
            viscous_alpha (optional): Background turbulent gas viscosity parameter. Defaults to zero.

        """

        # Should be zero, but sets up dispersion object at the same time
        # Thisin case self.calculate() has not been called yet.
        val = self.calculate(w, wave_number_x, wave_number_z, viscous_alpha)

        M = self.matrix_M(w)
        P = self.matrix_P(w)

        # B*vghat = 0, eigenvalues given by det(B)=0
        B = M + P

        # Take vgxhat = 1, calculate remaining gas velocity components
        Bprime = [[B[1,1], B[1,2]], [B[2,1], B[2,2]]]
        b = [-B[1,0], -B[2,0]]

        v = np.linalg.solve(Bprime, b)

        vg = [1.0, v[0], v[1]]

        # Gas density perturbation
        rhog = (self.kx*vg[0] + self.kz*vg[2])/(w - self.kx*self.vgx)

        A = self.matrix_inverse_A(w)
        AD = self.matrix_AD(A, w)

        # Matrix multiply to get dust velocity
        ux = lambda x: AD[0][0](x)*vg[0] + AD[0][1](x)*vg[1] + AD[0][2](x)*vg[2]
        uy = lambda x: AD[1][0](x)*vg[0] + AD[1][1](x)*vg[1] + AD[1][2](x)*vg[2]
        uz = lambda x: AD[2][0](x)*vg[0] + AD[2][1](x)*vg[1] + AD[2][2](x)*vg[2]

        # Size density perturbation
        sigma = lambda x: self.mu*self.size_density(x)*(self.kx*ux(x) + self.kz*uz(x))/(w - self.kx*self.ux(x))


        gas_cont = \
          self.kx*self.vgx*rhog + self.kx*vg[0] + self.kz*vg[2] - w*rhog
        f = lambda x: sigma(x)*(self.ux(x) - self.vgx)/self.mu/self.size_density(x)/x + (ux(x) - vg[0])/x
        gas_momx = (self.kx*self.vgx*vg[0] + self.kx*self.c**2*rhog + 2j*vg[1] + 1j*self.average(f) - w*vg[0])/w/vg[0]
        f = lambda x: sigma(x)*(self.uy(x) - self.vgy)/self.mu/self.size_density(x)/x + (uy(x) - vg[1])/x
        gas_momy = (self.kx*self.vgx*vg[1] - 0.5j*vg[0] + 1j*self.average(f) - w*vg[1])/w/vg[1]
        f = lambda x: (uz(x) - vg[2])/x
        gas_momz = (self.kx*self.vgx*vg[2] + self.kz*self.c**2*rhog + 1j*self.average(f) - w*vg[2])/w/vg[2]

        #print(gas_cont, gas_momx, gas_momy, gas_momz)

        #f = lambda x: self.kx*self.ux(x)*sigma(x) + self.mu*self.size_density(x)*(self.kx*ux(x) + self.kz*uz(x))

        #f = lambda x: self.kx*self.ux(x)*ux(x) + 2j*uy(x) - 1j*(ux(x) - vg[0])/x - 1j*rhog*(self.ux(x) - self.vgx)/x
        #f = lambda x: self.kx*self.ux(x)*uy(x) - 0.5j*ux(x) - 1j*(uy(x) - vg[1])/x - 1j*rhog*(self.uy(x) - self.vgy)/x
        #f = lambda x: self.kx*self.ux(x)*uz(x) - 1j*(uz(x) - vg[2])/x

        #tau = np.logspace(-3,-1,100)

        #plt.plot(tau, np.real(f(tau)))
        #plt.plot(tau, np.real(w*uz(tau)))
        #plt.plot(tau, np.imag(f(tau)))
        #plt.plot(tau, np.imag(w*sigma(tau)))

        #plt.xscale('log')
        #plt.show()

        return rhog, vg, sigma, [ux, uy, uz]
