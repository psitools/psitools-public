#
#  Just run pytest in the repo, and this will get picked up and run.
#
import numpy as np
import numpy.testing as npt
import pytest
from . import direct

# Tolerance for tests
rtol = 5e-7


@pytest.mark.mpi_skip()
def run_PL(testargs):
    ait = direct.Converger(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
def run_PB(testargs):
    ait = direct.ConvergerPowerBump(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
def run_LN(testargs):
    ait = direct.ConvergerLogNormal(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_doesitrun():
    ntaus = 25
    tsint = (1e-8, 1e-2)
    epstot = 3.0
    beta = -3.5
    alpha = 1e-8
    sigdiff = 1e-9
    dust_pressure = True
    Kx = Kz = 1e3
    i = np.arange(ntaus)+1
    taus = (-0.5*np.cos(np.pi*(2*i-1)/(2.0*ntaus))+0.5) \
           * (tsint[1]-tsint[0]) + tsint[0]
    ss = direct.StreamingSolver(taus,
                                        epstot=epstot,
                                        beta=beta,
                                        alpha=alpha,
                                        dust_pressure=dust_pressure,
                                        sigdiff=sigdiff)
    ss.build_system_matrix(Kx=Kx, Kz=Kz)
    ss.solve_eigen()


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PB_0(tmpdir):
    ans = run_PB({
            'tsint': (1e-08, 1.56),
            'epstot': 3,
            'beta': -3.5,
            'aL': 2.0/3.0*1.0,
            'aP': 1.0,
            'bumpfac': 2.0,
            'Kx': 4.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': None})
    #assert ans == 1
    ref = np.array((-0.029005533253048926+0.15419664580434422j,
                    -0.030041404593872712+0.15409922719380723j,
                    -0.03053511020161884+0.15404645287572324j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PB_1(tmpdir):
    ans = run_PB({
            'tsint': (1e-08, 1.56),
            'epstot': 3,
            'beta': -3.5,
            'aL': 2.0/3.0*1.0,
            'aP': 1.0,
            'bumpfac': 2.0,
            'Kx': 4.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': 4e-8,
            'dust_pressure': True})
    ref = np.array(
                   (0.1345493907813624-0.01672903776471424j,
                    0.13351780908580427-0.01698450431931421j,
                    0.1330210694554548-0.01710878302411626j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PB_2(tmpdir):
    ans = run_PB({
             'tsint': (1e-08, 1.56),
             'epstot': 3,
             'beta': -3.5,
             'aL': 2.0/3.0*1.0,
             'aP': 1.0,
             'bumpfac': 2.0,
             'Kx': 4.0,
             'Kz': 100.0,
             'refine': 3,
             'gridding': 'logarithmic',
             'prefix': tmpdir,
             'alpha': 4e-8})
    ref = np.array((0.01185500097834895+0.01382494694562451j,
                    0.011634542349422348+0.016713035662141162j,
                    0.012185844240649526+0.01725205057470446j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PL_0(tmpdir):
    # MRN tests
    ans = run_PL({
            'tsint': (1e-08, 1e-1),
            'epstot': 3.0,
            'beta': -3.5,
            'Kx': 80.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': 4e-8,
            'dust_pressure': True})
    ref = np.array(
                   (0.605001947246457+0.01855421363780353j,
                    0.6026448174624307+0.018214074745258046j,
                    0.6014473718398458+0.018048063756147426j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PL_1(tmpdir):
    ans = run_PL({
            'tsint': (1e-08, 1e-1),
            'epstot': 3.0,
            'beta': -3.5,
            'Kx': 80.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': None})
    ref = np.array((0.5512363292041821+0.11242857992948893j,
                    0.5489690582185277+0.11216071905699404j,
                    0.5478173144269354+0.11203395665778133j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_LN_0(tmpdir):
    ans = run_LN({
               'tsint': (1e-08, 1e-1),
               'epstot': 3.0,
               'sigma': 1e-2,
               'peak': 8e-2,
               'Kx': 100.0,
               'Kz': 100.0,
               'refine': 3,
               'gridding': 'chebyshevroots',
               'prefix': tmpdir,
               'alpha': 1e-7})
    ref = np.array(
                   (0.40950662474594923+0.2743694562314839j,
                    0.40944101911374947+0.2743957294399602j,
                    0.409424614329867+0.2744022809946226j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)
