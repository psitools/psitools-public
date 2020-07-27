#!/usr/bin/python

import numpy as np
import scipy as sp
import scipy.optimize as opt

import warnings


def unique_within_tol(a, tol=1e-12):
    if a.size == 0:
        return a
    b = a.copy()
    b.sort()
    d = np.append(True, np.diff(b))
    return b[d > tol]


class RationalApproximation():
    """Class calculating a rational function approximation
       given a set of samples.

     Use the AAA algorithm for rational approximation
     (Nakatsukasa et al. 2018) to approximate a function f(z) given a set of
     samples F at sample points Z.

    Args:
        domain: Domain for the rational approximation
        tol (optional): Requested tolerance, defaults to 1.0e-13
        clean_tol (optional): Tolerance for removing Froissart doublets. Defaults to 1.0e-10.
    """
    def __init__(self, domain, tol=1.0e-13, clean_tol=1.0e-10):
        self.domain = domain
        self.tol = tol
        self.clean_tol = clean_tol

    def calc_weights_residuals(self):
        """Calculate the weights and residuals"""

        # Elements that have not been included in the fit yet
        mF = np.ma.masked_array(self.F, mask=self.maskF).compressed()
        mZ = np.ma.masked_array(self.Z, mask=self.maskF).compressed()

        # Elements that have been included in the fit
        mf = np.ma.masked_array(self.f, mask=1-self.maskF).compressed()
        mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()

        # Construct Cauchy matrix
        m = len(mf)
        mCauchy = np.ndarray((self.M - m, m), dtype=np.complex128)
        for column in range(0, m):
            mCauchy[:, column] = 1/(mZ - mz[column])

        # Construct Loewner matrix A
        SF = np.diag(mF)
        Sf = np.diag(mf)
        A = np.matmul(SF, mCauchy) - np.matmul(mCauchy, Sf)

        # Singular value decomposition
        u, s, vh = np.linalg.svd(A, compute_uv=True)

        # New weights are in vector with lowest singular value
        self.weights = vh.T.conjugate()[:, len(mf)-1]

        # Calculate residuals at remaining points
        N = np.matmul(mCauchy, self.weights*mf)
        D = np.matmul(mCauchy, self.weights)
        self.R = np.abs(mF - N/D)

        # Return maximum residual relative to F
        return np.max(self.R)/np.max(np.abs(self.F))

    def first_step(self):
        """Select first two nodes and calculate weights"""
        # Choose first two points
        self.maskF[0] = 1
        self.maskF[1] = 1

        self.calc_weights_residuals()

    def step(self):
        """Select a new node and calculate weights."""
        try:
            # Mask point with highest residual
            i = np.argmax(self.R)
            # Need i-th element where maskF=0
            counts = np.cumsum(1 - self.maskF)
            i = np.searchsorted(counts, i + 1)
            self.maskF[i] = 1
            residuals = self.calc_weights_residuals()
        except np.linalg.LinAlgError as err:
            warnings.warn("LinAlgError occured in RationalApproximation." +
                          + "calc_weights_residuals: {}".format(str(err)))
            # Unset maskF as point was not used
            self.maskF[i] = 0
            # Mask point with second-highest residual
            i = np.argsort(self.R)[-2]
            # Need i-th element where maskF=0
            counts = np.cumsum(1 - self.maskF)
            i = np.searchsorted(counts, i + 1)
            self.maskF[i] = 1

            residuals = self.calc_weights_residuals()

        return residuals

    def calculate(self, F, Z):
        """Calculate rational approximation to given tolerance

        Args:
            F: Function samples
            Z: Sample points
        """

        # Check if F and Z have same shape
        if np.shape(F) != np.shape(Z):
            raise TypeError('Function samples and sample points need '
                            'to have the same shape')

        # Check F and Z are finite
        if np.isfinite(np.sum(F) + np.sum(Z)) == False:
            raise ValueError('Nan or Inf in sample points or function samples')

        self.F = F        # Function samples
        self.Z = Z        # Sample points
        self.M = len(F)   # Number of sample points

        self.f = F        # Fitted samples
        self.z = Z        # Points included in fit

        # Nodes that are part of the approximation will be masked out
        self.maskF = np.zeros(self.M)

        # Select the first two nodes
        self.first_step()

        # Add a maximum of nodes that is half the amount of sample points
        # Break if residual norm small enough.
        max_norm = 1.0e10
        for m in range(1, np.int(len(self.F)/2)):
            new_norm = self.step()
            if (new_norm < self.tol):
                break
            max_norm = new_norm

        # Clean up Froissart doublets
        n_cleanup = 0
        while self.cleanup() != 0:
            n_cleanup += 1

        # Maximum residual relative to F
        max_res = np.max(self.R)/np.max(np.abs(self.F))

        number_of_nodes_used = np.sum(self.maskF)

        return {"n_nodes" : number_of_nodes_used,
                "max_residual" : max_res}

    def find_zeros(self, method='arrowhead', secant_improve=True):
        """Find zeros of rational approximation"""

        # Compress to nodes used
        mf = np.ma.masked_array(self.f, mask=1-self.maskF).compressed()
        mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()

        if method == 'arrowhead':
            # Construct arrowhead matrix A
            d = np.insert(mz, 0, 0)
            A = np.diag(d)
            A[0, 1:] = self.weights*mf
            A[1:, 0] = np.ones(len(mz))

            B = np.diag(np.insert(np.ones(len(mz)), 0, 0))

            # Generalized eigenvalue problem
            e, v = sp.linalg.eig(A, B)

            # First two will be infinity, ignore
            poly_roots = np.atleast_1d(np.sort(e)[:-2])

        if method == 'polyroots':
            sf = self.weights*mf
            c = np.zeros(np.shape(sf), dtype=np.complex128)

            for i in range(0, len(mf)):
                roots_l = np.concatenate((mz[:i], mz[i+1:]))
                bj = np.polynomial.polynomial.polyfromroots(roots_l)
                c = c + sf[i]*bj

            poly_roots = np.atleast_1d(np.polynomial.polynomial.polyroots(c))

        # Sanity check: number of roots should be the number of nodes used
        if len(poly_roots) != len(self.weights) - 1:
            raise RuntimeError('Number of roots of rational approximation '
                               'not equal to number of nodes used')

        if secant_improve == True:
            # Do secant iteration to improve roots
            for i in range(0, len(poly_roots)):
                eps = 1.0e-4
                z0 = poly_roots[i]
                z1 = z0*(1 + eps)

                z1 += (eps if z1.real >= 0 else -eps)

                warnings.simplefilter('error', RuntimeWarning)
                try:
                    poly_roots[i] = opt.root_scalar(self.evaluate,
                                                    method='secant',
                                                    x0=z0, x1=z1).root
                except:
                    # If a warning occurs, automatically use the initial root
                    pass
                warnings.simplefilter('default', RuntimeWarning)

            # Different starting roots may end up on the same end root,
            # resulting in multiple equivalent roots. Return only unique roots.
            if len(poly_roots) > 1:
                poly_roots = unique_within_tol(poly_roots)

        return poly_roots


    def find_poles(self):
        """Find poles of rational approximation"""
        mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()

        # Construct arrowhead matrix A
        d = np.insert(mz, 0, 0)
        A = np.diag(d)
        A[0, 1:] = self.weights
        A[1:, 0] = np.ones(len(mz))

        B = np.diag(np.insert(np.ones(len(mz)), 0, 0))

        # Generalized eigenvalue problem
        e, v = sp.linalg.eig(A, B)

        # First two will be infinity, ignore
        e = e[2:]

        return e

    def find_poles_in_domain(self):
        """Find poles of rational approximation"""
        mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()

        # Construct arrowhead matrix A
        d = np.insert(mz, 0, 0)
        A = np.diag(d)
        A[0, 1:] = self.weights
        A[1:, 0] = np.ones(len(mz))

        B = np.diag(np.insert(np.ones(len(mz)), 0, 0))

        # Generalized eigenvalue problem
        e, v = sp.linalg.eig(A, B)

        # First two will be infinity, ignore
        e = e[2:]

        # Only consider poles inside domain
        sel = np.asarray(self.domain.is_in(e)).nonzero()

        return e[sel]

    def cleanup(self):
        """Attempt to clean up Froissart doublets.
        This is done by calculating the residuals of the poles of the rational
        approximation and removing the nearest node if the residual is smaller
        than a tolerance.
        """

        # Find the poles.
        poles = self.find_poles_in_domain()

        # Square contour around pole.
        L = 1.0e-5
        dz = L*np.exp(-0.25*1j*np.pi + np.arange(4)*0.5*np.pi*1j)

        # Do contour integral around poles
        remove_list = []
        for p in poles:
            z_int = p + dz
            f_int = self.evaluate(z_int)

            c_int = \
              0.5*(z_int[1] - z_int[0])*(f_int[1] + f_int[0]) + \
              0.5*(z_int[2] - z_int[1])*(f_int[2] + f_int[1]) + \
              0.5*(z_int[3] - z_int[2])*(f_int[3] + f_int[2]) + \
              0.5*(z_int[0] - z_int[3])*(f_int[0] + f_int[3])

            if (np.abs(c_int)/np.max(np.abs(self.F)) < self.clean_tol):
                # Remove closest point from interpolation
                mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()
                j = np.argmin(np.abs(p - mz))
                # Need j-th element where maskF= 1
                counts = np.cumsum(self.maskF)
                k = np.searchsorted(counts, j + 1)
                remove_list.append(k)

        # Remove points from mask
        for j in remove_list:
            self.maskF[j] = 0

        # Recalculate weights
        self.calc_weights_residuals()

        # Return number of modes removed
        return len(remove_list)

    def evaluate(self, z):
        """Evaluate rational approximation at point(s) z. Can only be done if
        the weights have been calculated.
        """

        # Make sure we can handle scalar and vector input.
        z = np.asarray(z)
        scalar_input = False
        if z.ndim == 0:
            z = z[None]  # Makes z 1D
            scalar_input = True
        else:
            original_shape = np.shape(z)
            z = np.ravel(z)

        ret = 0.0*z

        # Compress to nodes used
        mf = np.ma.masked_array(self.f, mask=1-self.maskF).compressed()
        mz = np.ma.masked_array(self.z, mask=1-self.maskF).compressed()

        # Calculate rational approximation from weights
        for i in range(0, len(z)):
            #d = self.weights/(z[i] - mz)
            #n = d*mf
            #ret[i] = np.sum(n)/np.sum(d)

            d = np.copy(self.weights)
            sel = np.asarray(self.weights != 0).nonzero()
            d[sel] = d[sel]/(z[i] - mz[sel])

            n = d*mf
            ret[i] = np.sum(n)/np.sum(d)

        if scalar_input:
            return np.squeeze(ret)

        return np.reshape(ret, original_shape)

class Circle():
    """Class representing a circular domain.

    Args:
        centre: centre of circle
        radius: radius of circle
    """
    def __init__(self, centre, radius):
        self.c = centre
        self.r = radius

    def generate_sample_points(self, n_sample):
        """Generate n_sample points, distributed randomly over the disc."""
        r = np.random.uniform(0, self.r, n_sample)
        phi = np.random.uniform(0, 2*np.pi, n_sample)
        return r*np.exp(1j*phi) + self.c

    def is_in(self, z):
        """Return True if z is strictly inside the circle, False otherwise"""
        return (np.abs(z - self.c) < self.r)

class Rectangle():
    """Class representing a rectangular domain.

    Args:
        x_range: (min_x, max_x)
        y_range: (min_y, max_y)
    """
    def __init__(self, x_range, y_range):
        self.xmin = x_range[0]
        self.xmax = x_range[1]
        self.ymin = y_range[0]
        self.ymax = y_range[1]

    def generate_random_sample_points(self, n_sample):
        """Generate n_sample points, distributed randomly over rectangle."""
        x = np.random.uniform(self.xmin, self.xmax, n_sample)
        y = np.random.uniform(self.ymin, self.ymax, n_sample)

        return x + 1j*y

    def is_in(self, z):
        """Return True if z is strictly inside the rectangle,
        False otherwise.
        """
        return np.where((np.real(z) > self.xmin) &
                        (np.real(z) < self.xmax) &
                        (np.imag(z) > self.ymin) &
                        (np.imag(z) < self.ymax), True, False)

    def grid_xy(self, N=100):
        """Return a uniform grid of the domain"""
        x = np.linspace(self.xmin, self.xmax, N)
        y = np.linspace(self.ymin, self.ymax, N)

        return x, y


class RootFollower():
    """Class for following the root of a function while slowly varying a
    parameter. Need func(z_start, k_start) = 0.
    """

    def __init__(self, func, z_start, k_start):
        self.func = func
        self.z_start = z_start
        self.k_start = k_start

    def calculate(self, k_want):
        """Calculate the root of func(z, k_want)"""
        k_want = np.atleast_1d(k_want)
        ret = np.zeros((len(k_want)), dtype=np.complex128)

        sel = np.asarray(k_want < self.k_start).nonzero()
        ret[sel] = self._calculate(k_want[sel])

        sel = np.asarray(k_want > self.k_start).nonzero()
        ret[sel] = self._calculate(k_want[sel])

        return ret

    def _calculate(self, k_want):
        # Assume k_want is sorted and all < k_start
        #k_want = np.atleast_1d(k_want)
        ret = np.zeros((len(k_want)), dtype=np.complex128)

        k = self.k_start
        z0 = self.z_start

        # Secant method needs second point.
        # There appears to be a bug in scipy in automatically adding x1 for
        # complex x0.
        eps = 1.0e-4
        z1 = z0*(1 + eps)
        z1 += (eps if z1.real >= 0 else -eps)

        for i in range(0, len(k_want)):
            idx = i
            if (k_want[0] < self.k_start):
                idx = -(i+1)

            kw = k_want[idx]
            k_step = kw - k

            finished = False
            #print('kw = ', kw)
            while (True):
                if (finished is True):
                    break

                print("Current k:", k)
                while (True):
                    print("Trying to find root at k = ", k + k_step, k_step)

                    f = lambda x: self.func(x, k + k_step)
                    z = opt.root_scalar(f, method='secant',
                                        x0=z0, x1=z1,
                                        maxiter=10)

                    if (z.converged is True):
                        if z.root.imag*z0.imag > 0:
                            print("Converged at z = ", z.root)
                            break

                    #z0 = z1
                    #z1 = z.root

                    k_step = k_step/10
                    if (k_step < 1.0e-6):
                        return ret

                k = k + k_step
                k_step = k_step*np.sqrt(2)

                if (k_step > 0):
                    if (k + k_step > kw):
                        k_step = kw - k
                        finished = True
                else:
                    if (k + k_step < kw):
                        k_step = kw - k
                        finished = True

            ret[idx] = z.root

        return ret

class ClosedPath():
    """Class containing a piecewise linear closed contour in the complex plane.

    The main purpose is to be able to count the roots of a function inside the contour.

    Args:
        f: function of a complex variable f(z)
        z_list: list of complex values specifying the piecewise linear contour, in counterclockwise orientation.
    """
    def __init__(self, f, z_list):
        self.func = f
        self.points = np.asarray(z_list)
        self.f_points = self.func(z_list)

        self.centre = np.mean(self.points)

    def sum_angles(self):
        """Sum over argument ratio, should equal number of enclosed roots"""

        # Try and avoid division by zero
        f = self.f_points + 1.0e-30*1j
        return 0.5*np.sum(np.angle(np.roll(f, -1)/f))/np.pi

    def interweave(self, a, b):
        """Interweave two arrays a and b, starting with a[0]"""
        c = np.empty((a.size + b.size,), dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b

        return c

    def refine(self):
        """Double the number of points across the contour"""
        z_add = 0.5*(self.points + np.roll(self.points, -1))
        f_add = self.func(z_add)

        self.points = self.interweave(self.points, z_add)
        self.f_points = self.interweave(self.f_points, f_add)

    def refine_select(self, tol=0.1, max_step=0.1):
        """Refine where argument jump larger than tolerance

        Args:
            tol (optional): tolerance of argument difference between neighbouring points. Defaults to 0.1.
            max_step (optional): maximum distance between neighbouring points. Defaults to 0.1.
        """
        # Potentially add points in the middle of intervals
        z_add = 0.5*(self.points + np.roll(self.points, 1))

        # Argument difference between neighbouring points
        da = np.angle(self.f_points) - np.angle(np.roll(self.f_points, 1))
        # Distance between neighbouring points
        dz = np.abs(self.points - np.roll(self.points, 1))

        # Make sure the distance between points does not change too rapidly
        step_change_too_large = \
          ((dz > 2*np.roll(dz, 1)) | (dz > 2*np.roll(dz, -1)))

        # Want to add points if angle difference too large, step change is
        # too large, or step size too large
        want_refine = ((np.abs(da) > tol) |
                       (step_change_too_large)) | (dz > max_step)

        # Select points to add, never making the step size too small
        sel = np.asarray((want_refine) & (dz > 1.0e-8)).nonzero()

        # Add points and calculate new function values
        z_add = z_add[sel[0]]
        f_add = self.func(z_add)

        if np.all(np.isfinite(f_add)) != True:
            raise ValueError("Infinite value in f_add")

        # Insert into arrays
        self.points = np.insert(self.points, sel[0], z_add)
        self.f_points = np.insert(self.f_points, sel[0], f_add)

        # Return number of points added
        return len(sel[0])

    def count_roots(self, verbose=False, max_iter=100, tol=0.1, max_step=0.1):
        """Count number of roots in domain

        Args:
            verbose (bool, optional): Flag whether to print progress. Defaults to False
            max_iter (optional): Maximum number of refinement iterations. Defaults to 100. RuntimeError if not converged before.
            tol (optional): Tolerance in jump in argument. Defaults to 0.1.
            max_step (optional): Maximum step size along contour. Defaults to 0.1,
        """

        self.success = False
        for i in range(0, max_iter):
            # Refine where necessary, count number of terms added
            n_add = self.refine_select(tol=tol, max_step=max_step)

            if verbose == True:
                print(i, self.sum_angles(), n_add)

            # Nothing added; done
            if n_add == 0:
                a = self.sum_angles()
                ret = np.rint(a)
                if np.abs(a - ret) < 1.0e-6:
                    self.success = True
                    return int(ret)

                raise RuntimeError("No convergence to integer number of roots")

        raise RuntimeError("Maximum number of iterations exceeded")

        return []
