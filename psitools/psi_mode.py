#!/usr/bin/python

import numpy as np
import scipy.optimize as opt
import warnings
try:
    import matplotlib.pyplot as plt
except ImportError as error:
    # as of python 3.6 now throws ModuleNotFoundError
    print('Will run despite error importing matplotlib:', error)

from . import psi_dispersion as psid
from . import complex_roots as cr


class PSIMode():
    """Class for calculating PSI modes.

    Args:
        dust_to_gas_ratio: Dust to gas ratio
        stokes_range: Range of Stokes numbers to consider
        real_range: Range on real axis to consider
        imag_range: Range on imaginary axis to consider
        sound_speed_over_eta (optional) : Gas sound speed divided by eta. Defaults to 1/0.05.
        size_distribution_power (optional): Power law index of size distribution. Defaults to 3.5 = MRN. If size_density is not None, size_distribution_power is ignored.
        single_size_flag (optional): If true, calculate for single size at maximum Stokes number.
        n_sample (optional): Number of sample points for the dispersion relation. Defaults to 20.
        max_zoom_domains (optional): Number of times to zoom in on closest growing mode if no growth is found. Defaults to 1.
        verbose_flag (optional): If True, print more info to screen. Defaults to False.
        size_density (optional): Instance of SizeDensity class. Defaults to None, in which case a power law size density is constructed from size_distribution_power.
        tol (optional): Tolerance in rational function approximation. Defaults to 1.0e-13.
        clean_tol (optional): Tolerance for cleaning up fake poles and roots in rational function approximation. Defaults to 1.0e-4.
        max_secant_iterations (optional): Maximum number of iterations searching for the root of the exact dispersion relation. Defaults to 100.
    """
    def __init__(self,
                 dust_to_gas_ratio,
                 stokes_range,
                 real_range,
                 imag_range,
                 sound_speed_over_eta=1/0.05,
                 size_distribution_power=3.5,
                 single_size_flag=False,
                 n_sample=20,
                 max_zoom_domains=1,
                 verbose_flag=False,
                 size_density=None,
                 tol=1.0e-13,
                 clean_tol=1.0e-4,
                 max_secant_iterations=100):
        # If dispersion relation is given by f(w)=0, this function calculates
        # the function f(w).
        self.dispersion = psid.PSIDispersion(dust_to_gas_ratio,
                                             stokes_range,
                                             sound_speed_over_eta,
                                             size_distribution_power,
                                             single_size_flag,
                                             size_density).calculate
        # Create rectangular domain
        self.domain = cr.Rectangle(real_range, imag_range)
        self.n_sample = n_sample
        self.max_zoom_level = max_zoom_domains
        self.max_zoom_domains_per_level = 4
        self.max_secant_iterations = max_secant_iterations

        self.verbose_flag = verbose_flag

        self.max_convergence_iterations = 0
        self.minimum_interpolation_nodes = 4
        self.force_at_least_one_root = True

        # AAA rational approximation
        self.ra = cr.RationalApproximation(self.domain,
                                           tol=tol,
                                           clean_tol=clean_tol)

    def calculate(self, wave_number_x, wave_number_z, viscous_alpha=0,
                  guess_roots=[]):
        """Calculate complex mode frequency at wave number Kx and Kz and viscosity parameter viscous_alpha.

        Args:
            wave_number_x: Nondimensional (using YG05) x wave number
            wave_number_z: Nondimensional (using YG05) z wave number
            viscous_alpha (optional): Background turbulent gas viscosity parameter. Defaults to zero.
        """

        # Exact dispersion relation evaluated at wave numbers and diffusion
        self.disp = lambda z: self.dispersion(z,
                                              wave_number_x,
                                              wave_number_z,
                                              viscous_alpha)

        # Calculate sample points and sample function values
        self.z_sample = self.domain.generate_random_sample_points(self.n_sample)
        self.n_function_call = self.n_sample
        self.f_sample = self.disp(self.z_sample)

        # Put zoom domain around guessed roots
        for centre in guess_roots:
#            self.add_extra_domain(extra_domain_size=[0.001, 0.001],
#                                  centre=centre)
           # If the domain is too small it is hard to track the 1e-6
           # growth of secular modes near the axis.
           domain_size = min((1e-3, np.abs(centre.imag)*4))
           self.add_extra_domain(extra_domain_size=[domain_size, domain_size],
                                  centre=centre)

        for n in range(0, self.max_zoom_level + 1):
            # Calculate rational approximation
            self.ra.calculate(self.f_sample, self.z_sample)

            # Find the zeros of the rational approximation.
            zeros = self.ra.find_zeros()
            sel = np.asarray(self.domain.is_in(zeros)).nonzero()
            zeros = zeros[sel]

            # Reduce clean tolerance until enough nodes and at least one root
            min_nodes = self.minimum_interpolation_nodes
            fac = 1 + n
            if len(guess_roots) > 0:
                fac = fac + 1
            min_nodes *= (1 + n + len(guess_roots))
            old_clean_tol = self.ra.clean_tol
            while (np.sum(self.ra.maskF) < min_nodes or
                   (len(zeros) == 0 and self.force_at_least_one_root == True)):
                self.ra.clean_tol = 0.5*self.ra.clean_tol

                # Stop if too small, nothing to be done at this point
                if self.ra.clean_tol < self.ra.tol:
                    break

                self.log_print('Reducing clean_tol to '
                               '{}'.format(self.ra.clean_tol))
                self.ra.calculate(self.f_sample, self.z_sample)

                # Find the zeros of the rational approximation.
                zeros = self.ra.find_zeros()
                sel = np.asarray(self.domain.is_in(zeros)).nonzero()
                zeros = zeros[sel]

            self.ra.clean_tol = old_clean_tol

            n_nodes = np.sum(self.ra.maskF)
            self.log_print('Number of nodes used: {}'.format(n_nodes))

            # Find the zeros of the rational approximation.
            zeros = self.ra.find_zeros()
            self.log_print('All zeros: {}'.format(zeros))

            # Select roots to zoom into potentially
            sel = np.asarray((np.imag(zeros) < self.domain.ymax) &
                             (np.real(zeros) < self.domain.xmax) &
                             (np.real(zeros) > self.domain.xmin)).nonzero()
            pot_zoom_roots = zeros[sel]

            # Size of zoom domain in x (= real part)
            zoom_domain_size_x = np.power(10.0, -1 - 0.5*n)

            if len(pot_zoom_roots) > 0:
                # Sort according to imaginary part
                pot_zoom_roots = \
                  pot_zoom_roots[np.argsort(np.imag(pot_zoom_roots))]

                zoom_roots = [pot_zoom_roots[-1]]
                for i in range(1, len(pot_zoom_roots)):
                    z = pot_zoom_roots[-1-i]
                    add_flag = True
                    for j in range(0, len(zoom_roots)):
                        if (np.abs(np.real(z) - np.real(zoom_roots[j])) <
                            zoom_domain_size_x):
                            add_flag = False
                    if add_flag == True:
                        zoom_roots.append(z)

                # Limit number of domains to add every level
                zoom_roots = zoom_roots[:self.max_zoom_domains_per_level]

                # Start secant iteration from this root if no growing roots
                maximum_growing_zero = zoom_roots[0]

                ## # Start secant iteration from this root if no growing roots
                ## maximum_growing_zero = zoom_roots[zoom_roots.imag.argmax()]
                ## # Select roots that are *almost* growing as zoom sites
                ## sel = np.asarray(np.imag(zoom_roots) >
                ##                  -2*np.abs(np.imag(maximum_growing_zero)))
                ## zoom_roots = zoom_roots[sel]

                ## # Make sure we always have a zoom domain
                ## if len(zoom_roots) == 0:
                ##     zoom_roots = np.asarray([maximum_growing_zero])

                ## # If a growing mode is found, only zoom in on growing mode
                ## if (n > 0 and
                ##     len(zoom_roots) > 1):
                ##     # Sort according to imaginary part
                ##     zoom_roots = zoom_roots[np.argsort(np.imag(zoom_roots))]
                ##     n_zoom = np.sum(np.asarray(zoom_roots.imag > 0))
                ##     if (n_zoom > 0 and
                ##         n_zoom < len(zoom_roots) - 1):
                ##         zoom_roots = zoom_roots[(-n_zoom-1):]


            else:
                # Nothing to zoom into or to start searching from
                maximum_growing_zero = None
                zoom_roots = pot_zoom_roots

            # Look at zeros inside domain
            sel = np.asarray(self.domain.is_in(zeros)).nonzero()
            zeros = np.asarray(zeros[sel])
            self.log_print('All zeros in domain: {}'.format(zeros))

            # For multiple zoom levels, zoom in only on actual zeros
            #if n > 0:
            #    zoom_roots = np.copy(zeros)

            max_iter = self.max_secant_iterations
            # If we can still zoom, limit iterations
            if n < self.max_zoom_level:
                max_iter = self.n_sample
            # If no zeros in domain, start from least damped root
            if (len(zeros) == 0 and
                maximum_growing_zero is not None):
                zeros = np.asarray([maximum_growing_zero])
                max_iter = 5

            # Find the roots of the dispersion relation
            ret = self.find_dispersion_roots(zeros, max_iter)

            # Quit if a root found or no more zoom levels or sites
            if (len(ret) > 0 or
                n == self.max_zoom_level or
                len(zoom_roots) == 0):
                return ret

            # Not root found: add zoom domains
            for zoom_centre in zoom_roots:
                # Limit y size to imaginary part of centre
                sy = np.min([0.1, np.abs(np.imag(zoom_centre))])

                sz = [zoom_domain_size_x, sy]
                self.add_extra_domain(extra_domain_size=sz,
                                      centre=zoom_centre)

        return ret

    def log_print(self, arg):
        """Only print info if verbose_flag is True"""
        if (self.verbose_flag == True):
            print(arg)

    def find_dispersion_roots(self, zeros, max_iter):
        # zeros = roots of rational approximation.
        zeros = np.atleast_1d(zeros)
        for i in range(0, len(zeros)):
            self.log_print('Starting iteration at z = {}'.format(zeros[i]))

            # Secant iteration starting either from approximate zero or minimum
            zeros[i] = self.find_root(zeros[i], max_iter=max_iter)

        # Return only zeros inside domain
        sel = np.asarray(self.domain.is_in(zeros)).nonzero()
        return zeros[sel]

    def add_extra_domain(self, extra_domain_size=[0.1, 0.1], centre=0.0):
        """Add extra sample points in domain close to real axis"""

        #original_centre = centre

        # Make sure whole zoom domain is inside original domain
        if np.real(centre) < self.domain.xmin + 0.5*extra_domain_size[0]:
            centre = self.domain.xmin + 0.5*extra_domain_size[0] + \
              1j*np.imag(centre)
        if np.real(centre) > self.domain.xmax - 0.5*extra_domain_size[0]:
            centre = self.domain.xmax - 0.5*extra_domain_size[0] + \
              1j*np.imag(centre)
        if np.imag(centre) < self.domain.ymin + 0.5*extra_domain_size[1]:
            centre = np.real(centre) + 1j*(self.domain.ymin +
                                           0.5*extra_domain_size[1])
        if np.imag(centre) > self.domain.ymax - 0.5*extra_domain_size[1]:
            centre = np.real(centre) + 1j*(self.domain.ymax -
                                           0.5*extra_domain_size[1])

        self.log_print('Adding zoom domain around {}'.format(centre))

        # Number of extra points to add
        extra_n_sample = self.n_sample

        # Create zoom domain around potential mode
        real_range = [np.real(centre) - 0.5*extra_domain_size[0],
                      np.real(centre) + 0.5*extra_domain_size[0]]
        imag_range = [np.imag(centre) - 0.5*extra_domain_size[1],
                      np.imag(centre) + 0.5*extra_domain_size[1]]
        extra_domain = cr.Rectangle(real_range, imag_range)

        # Generate sample points
        extra_z_sample = \
          extra_domain.generate_random_sample_points(extra_n_sample)

        #if extra_domain.is_in(original_centre):
        #    extra_z_sample[0] = original_centre

        extra_f_sample = self.disp(extra_z_sample)

        # Add to original arrays
        self.z_sample = np.concatenate((self.z_sample, extra_z_sample))
        self.f_sample = np.concatenate((self.f_sample, extra_f_sample))
        self.n_function_call += extra_n_sample

    def find_minimum(self):
        """Find location of minimum of rational approximation in domain"""
        f = lambda x: np.abs(self.ra.evaluate(x[0] + 1j*x[1]))

        bounds = ((self.domain.xmin, self.domain.xmax),
                  (self.domain.ymin, self.domain.ymax))
        ret = opt.differential_evolution(f, bounds)

        self.log_print(ret)

        return ret.x[0] + 1j*ret.x[1]

    def find_root(self, w0, max_iter=1000):
        """Find a root of the dispersion relation, starting from initial guess w0"""
        # Secant method needs second point.
        # There appears to be a bug in scipy in automatically adding x1 for
        # complex x0.
        eps = 1.0e-6
        w1 = w0 * (1 + eps)
        w1 += (eps if w1.real >= 0 else -eps)

        # Experiment: do a maximum of 5 iterations
        # Check if solution runs away from initial point
        # If not, continue for as long as it takes
        z = opt.root_scalar(self.disp, method='secant',
                            x0=w0, x1=w1,
                            maxiter=5)
        self.n_function_call += z.function_calls
        self.log_print("Result after 5 iterations:")
        self.log_print(z)
        self.log_print("|z - z0|/|z0| = {}".format(np.abs(z.root - w0)/np.abs(w0)))

        if (z.converged == False and
            np.abs(z.root - w0)/np.abs(w0) < 1 and
            max_iter > 5):
            self.log_print("Starting further iteration")

            # Start where we left off, again need second point
            w0 = z.root
            w1 = w0 * (1 + eps)
            w1 += (eps if w1.real >= 0 else -eps)

            z = opt.root_scalar(self.disp, method='secant',
                                x0=w0, x1=w1,
                                maxiter=max_iter)
            self.n_function_call += z.function_calls
            self.log_print(z)
        else:
            if (z.converged == False):
                self.log_print("Secant is running away; no further iterations")

        if (z.converged == True and
            self.domain.is_in(z.root) == True):
            self.max_convergence_iterations = \
              np.max([self.max_convergence_iterations, z.iterations])
            return z.root
        # Return value that will be deselected (TODO)
        return -1j

    def plot_dispersion(self, wave_number_x, wave_number_z,
                        viscous_alpha=0,
                        N=100, show_exact=False, x=None, y=None):
        """Plot the approximate dispersion relation over the whole domain, and, possibly, the exact dispersion relation to compare (*very* expensive).

        Args:
            wave_number_x: Nondimensional (using YG05) x wave number
            wave_number_z: Nondimensional (using YG05) z wave number
            viscous_alpha (optional): Background turbulent gas viscosity parameter. Defaults to zero.
            N (optional): Numer of points to plot in x and y, defaults to 100
            show_exact (optional): If True, also plot the exact dispersion relation (*very* expensive). Defaults to False.
            x (optional): Array of x values to plot. Defaults to None, in which case the whole x domain is shown with N equidistant points.
            y (optional): Array of y values to plot. Defaults to None, in which case the whole y domain is shown with N equidistant points.
        """
        if (x is None or y is None):
            x, y = self.domain.grid_xy(N=N)
        xv, yv = np.meshgrid(x, y)
        z = xv + 1j*yv

        # Rational approximation
        f_appro = self.ra.evaluate(z)

        if show_exact == True:
            f_exact = self.dispersion(z, wave_number_x, wave_number_z,
                                      viscous_alpha=viscous_alpha)
            plt.subplot(122)
            plt.contourf(x, y, np.log10(np.abs(f_exact)), 20)
            plt.colorbar()
            plt.subplot(121)

        plt.contourf(x, y, np.log10(np.abs(f_appro)), 20)
        plt.colorbar()

        x_node = np.real(self.z_sample)
        y_node = np.imag(self.z_sample)
        sel = np.where((x_node > x[0]) &
                       (x_node < x[-1]) &
                       (y_node > y[0]) &
                       (y_node < y[-1]), True, False)
        x_node = x_node[sel]
        y_node = y_node[sel]

        plt.plot(x_node, y_node, linestyle='', marker='.', color='black')

        mz = np.ma.masked_array(self.z_sample,
                                mask=1-self.ra.maskF).compressed()
        x_node = np.real(mz)
        y_node = np.imag(mz)
        sel = np.where((x_node > x[0]) &
                       (x_node < x[-1]) &
                       (y_node > y[0]) &
                       (y_node < y[-1]), True, False)
        x_node = x_node[sel]
        y_node = y_node[sel]

        plt.plot(x_node, y_node, linestyle='', marker='.', color='red')

        plt.show()
