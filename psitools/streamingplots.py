import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
plt.rcParams['axes.formatter.limits'] = (-2, 2)
import streamingtools
import h5py
import scipy.special


def plotFastestEigenmode(ss):
    fig = plt.figure(figsize=(8,12))
    axs = fig.subplots(nrows=4, ncols=2)
    axs[0][0].set_title('Growing Part')
    axs[0][1].set_title('Oscillating Part')
    for ii,key in enumerate(ss.fastest_eigenvec_by_taus.dtype.names):
        axs[ii][0].plot(ss.taus, ss.fastest_eigenvec_by_taus[key].real,color='C0')
        axs[ii][1].plot(ss.taus, ss.fastest_eigenvec_by_taus[key].imag,color='C1')
        axs[ii][0].set_ylabel(key)
    for ax in axs.flatten():
        ax.set_xlabel(r'$\tau_s$')
        ax.set_xlim((ss.taus.min(),ss.taus.max()))
    fig.suptitle('Dust Fluid Eigenfunctions');
    fig.tight_layout()
    return fig


class GridRunBase:
    # a global that sets the shrink of colorbars
    shrinkfac = 1.0

    def __init__(self, name, prefix=''):
        self.name = name
        self.vmin = 1e-8 # min for colorscales

    def get_RichConvMesh(self):
        '''Richardson extrapolation from backtraced eigenvalues, overridden by convergence'''
        richconvmesh = np.zeros((len(self.Kzaxis),len(self.Kxaxis)))
        for iz in range(0,len(self.Kzaxis)):
            for ix in range(0,len(self.Kxaxis)):
                #low, high = self.backfastesteigens[-2:,iz,ix]
                low, high = self.fastesteigens[-2:,iz,ix]
                if convcriteria(low,high):
                    richconvmesh[iz,ix] = high.imag
                else:
                    #richconvmesh[iz,ix] = richardson(self.backfastesteigens[:,iz,ix].imag)[0]
                    richconvmesh[iz,ix] = richardson(self.fastesteigens[:,iz,ix].imag)[0]
        return richconvmesh

    def plotRichConvMesh(self, log=False):
        return self._plotAccelConvMesh(self.get_RichConvMesh(), log=log)

    def plotAitkenConvMesh(self, log=False):
        '''Aitken extrapolation from backtraced eigenvalues, overridden by convergence'''
        aitkenconvmesh = np.zeros((len(self.Kzaxis),len(self.Kxaxis)))
        for iz in range(0,len(self.Kzaxis)):
            for ix in range(0,len(self.Kxaxis)):
                low, high = self.backfastesteigens[-2:,iz,ix]
                if convcriteria(low,high):
                    aitkenconvmesh[iz,ix] = high.imag
                else:
                    aitkenconvmesh[iz,ix] = aitkenize(self.backfastesteigens[:,iz,ix].imag)[-1]
        return self._plotAccelConvMesh(aitkenconvmesh, log=log)

    def _plotAccelConvMesh(self, accelconvmesh, log=False):
        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), accelconvmesh, norm=matplotlib.colors.LogNorm(vmin=self.vmin))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), accelconvmesh)
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Extrapolated Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.fastesteigens[-1,:,:].real)
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig


    def plotAitkenMesh(self, level=-1, log=False):
        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.aitkengrowthrates[level,:,:], norm=matplotlib.colors.LogNorm(vmin=self.vmin))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.aitkengrowthrates[level,:,:])
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Extrapolated Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.aitkenoscillate[level,:,:])
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig

    def plotAitkenMeshTol(self, reltol, level=-1, log=False):
        """ reltol is relative the max of the fastest un-accelerated value , but relative the local value in oscillating part"""

        cutoff = reltol*(self.fastesteigens[level,:,:].imag).max() #always positive in practice
        errmap = np.abs(self.aitkengrowthrates[level,:,:]-self.aitkengrowthrates[level-1,:,:])
        maskedgrowth = np.ma.MaskedArray(self.aitkengrowthrates[level,:,:], mask=(errmap>cutoff))

        cutoff = reltol*(np.abs(self.fastesteigens[level,:,:].real))
        errmap = np.abs(self.aitkenoscillate[level,:,:]-self.aitkenoscillate[level-1,:,:])
        maskedoscillate = np.ma.MaskedArray(self.aitkenoscillate[level,:,:], mask=(errmap>cutoff))

        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10(maskedgrowth))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), maskedgrowth)
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Extrapolated Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(np.log10(logedges(self.Kxaxis)), np.log10(logedges(self.Kzaxis)), maskedoscillate)
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig



    def plotFastestMesh(self, level=-1, log=False):
        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10((self.fastesteigens[level,:,:]).imag))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), (self.fastesteigens[level,:,:]).imag)
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.fastesteigens[level,:,:].real)
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig


    def plotFastestMeshTol(self, reltol, level=-1, log=False):

        cutoff = reltol*(self.fastesteigens[level,:,:].imag).max() #always positive in practice
        errmap = np.abs(self.fastesteigens[level,:,:].imag-self.fastesteigens[level-1,:,:].imag)
        maskedgrowth = np.ma.MaskedArray(self.fastesteigens[level,:,:].imag, mask=(errmap>cutoff))

        cutoff = reltol*np.abs(self.fastesteigens[level,:,:].real)
        errmap = np.abs(self.fastesteigens[level,:,:].real-self.fastesteigens[level-1,:,:].real)
        maskedoscillate = np.ma.MaskedArray(self.fastesteigens[level,:,:].real, mask=(errmap>cutoff))

        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10(maskedgrowth))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), maskedgrowth)
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(np.log10(logedges(self.Kxaxis)), np.log10(logedges(self.Kzaxis)), maskedoscillate)
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig



    def plotFastestRelMesh(self, level=-1):
        fig = plt.figure(figsize=(12,4))
        axs = fig.subplots(ncols=4,nrows=1)

        cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), (self.fastesteigens[level,:,:]).imag)
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Growth rate $[\Omega^{-1}]$')

        relchange =  abs((self.fastesteigens[level,:,:]).imag-(self.fastesteigens[level-1,:,:]).imag)/((self.fastesteigens[level,:,:].imag).max())
        cm = axs[1].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10(relchange))
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Change in Growth Rate')

        cm = axs[2].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.fastesteigens[level,:,:].real)
        cb = plt.colorbar(cm, ax = axs[2], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Oscillatory part of eigenvalue $[\Omega^{-1}]$')

        relchange =  abs((self.fastesteigens[level,:,:]).real-(self.fastesteigens[level-1,:,:]).real)/(abs(self.fastesteigens[level,:,:].real).max())
        cm = axs[3].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10(relchange))
        cb = plt.colorbar(cm, ax = axs[3], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Change in Oscillatory part')

        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        plt.tight_layout()
        return fig


    def plotAitkenFastestMesh(self, level=-1, log=False):
        fig = plt.figure(figsize=(10,5))
        axs = fig.subplots(ncols=2,nrows=1)
        if log:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), np.log10(self.aitkengrowthrates[level,:,:].clip(1e-8,1e100)))
        else:
            cm = axs[0].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), self.aitkengrowthrates[level,:,:])
        cb = plt.colorbar(cm, ax = axs[0], orientation='horizontal', shrink=self.shrinkfac)
        cb.set_label(r'Extrapolated Growth rate $[\Omega^{-1}]$')

        cm = axs[1].pcolormesh(logedges(self.Kxaxis), logedges(self.Kzaxis), (self.fastesteigens[level,:,:]).imag)
        cb = plt.colorbar(cm, ax = axs[1], orientation='horizontal', extend='neither', shrink=self.shrinkfac);
        cb.set_label(r'Plain Growth Rate $[\Omega^{-1}]$')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$K_x$')
            ax.set_ylabel(r'$K_z$')
            ax.set_aspect('equal')
        return fig



class GridRun(GridRunBase):

    def __init__(self, name, prefix=''):
        super().__init__(name, prefix=prefix)
        self.name = name

        with h5py.File(os.path.join(prefix,name+'.hdf5'),'r') as hf:
            if 'Kxaxis' in list(hf[name].keys()):
                self.Kxaxis = hf[name+'/Kxaxis'][...]
            if 'Kzaxis' in list(hf[name].keys()):
                self.Kzaxis = hf[name+'/Kzaxis'][...]
            if 'Kxgrid' in list(hf[name].keys()):
                self.Kxgrid = hf[name+'/Kxgrid'][...]
            if 'Kzgrid' in list(hf[name].keys()):
                self.Kzgrid = hf[name+'/Kzgrid'][...]
            if 'aitkengrowthrates' in list(hf[name].keys()):
                self.aitkengrowthrates = hf[name+'/aitkengrowthrates'][...]
            if 'aitkenoscillate' in list(hf[name].keys()):
                self.aitkenoscillate = hf[name+'/aitkenoscillate'][...]
            self.fastesteigens = hf[name+'/fastesteigens'][...]
            if 'backfastesteigens' in list(hf[name].keys()):
                self.backfastesteigens = hf[name+'/backfastesteigens'][...]
            self.beta = hf[name].attrs['beta']
            self.tsint = hf[name].attrs['tsint']
            self.epstot = hf[name].attrs['epstot']
            if 'alleigens' in list(hf[name].keys()):
                self.alleigens = []
                for iz, kz in enumerate(self.Kzaxis):
                    alleigensz = []
                    for ix, kx in enumerate(self.Kxaxis):
                        alleigensx = []
                        for ik,key in enumerate(list(hf[name+'/alleigens/'+str(iz)+'/'+str(ix)].keys())):
                            alleigensx.append(hf[name+'/alleigens/'+str(iz)+'/'+str(ix)+'/'+str(ik)])
                        alleigensz.append(alleigensx)
                    self.alleigens.append(alleigensz)



    def reSimMode(self, Kx, Kz, ntaus=257, gridding='logarithmic'):
        taus = streamingtools.get_gridding(streamingtools.gridmap[gridding], self.tsint, ntaus)

        ss = streamingtools.StreamingSolver(taus, epstot=self.epstot, beta=self.beta)
        ss.build_system_matrix(Kx, Kz)
        ss.solve_eigen()
        return ss

    def add_figtitle(self, fig):
        fig.suptitle(r'$\tau =$ [{:4.2e}, {:4.2e}] $\beta=${:4.1f} $\epsilon = ${:4.2e}'.format(self.tsint[0], self.tsint[1], self.beta, self.epstot));

    def plotRichConvMesh(self, log=False):
        fig = super().plotRichConvMesh(log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenConvMesh(self, log=False):
        fig = super().plotAitkenConvMesh(log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenMesh(self, level=-1, log=False):
        fig = super().plotAitkenMesh(level=level,log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenMeshTol(self, reltol, level=-1, log=False):
        fig = super().plotAitkenMeshTol(reltol, level=level, log=log)
        self.add_figtitle(fig)
        return fig

    def plotFastestMesh(self, level=-1, log=False):
        fig = super().plotFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig

    def plotFastestMeshTol(self, reltol, level=-1, log=False):
        fig = super().plotFastestMeshTol(reltol, level=level)
        self.add_figtitle(fig)
        return fig

    def plotFastestRelMesh(self, level=-1):
        fig = super().plotFastestRelMesh(level=level)
        self.add_figtitle(fig)
        plt.tight_layout()
        return fig


    def plotAitkenFastestMesh(self, level=-1, log=False):
        fig = super().plotAitkenFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig

    def plotMaxFastestEigenmode(self, refine=9, level=-1):
        mandex = (np.argmax((-self.fastesteigens[-1,:,:].real).max(axis=1)), np.argmax((-self.fastesteigens[-1,:,:].real).max(axis=0)))
        ss = reSimMode(Kx=self.Kxaxis[mandex[1]], Kz=self.Kzaxis[mandex[0]], taus=2**refine+1)
        print('Fastest growing eigenmode in new calculation is ',ss.get_fastest_growth())
        plotFastestEigenmode(ss)


class LogNormGridRun(GridRunBase):

    def __init__(self, name, prefix=''):
        self.name = name

        with h5py.File(os.path.join(prefix,name+'.hdf5'),'r') as hf:
            self.Kxaxis = hf[name+'/Kxaxis'][...]
            self.Kzaxis = hf[name+'/Kzaxis'][...]
            self.aitkengrowthrates = hf[name+'/aitkengrowthrates'][...]
            self.aitkenoscillate = hf[name+'/aitkenoscillate'][...]
            self.fastesteigens = hf[name+'/fastesteigens'][...]
            self.sigma = hf[name].attrs['sigma']
            self.peak = hf[name].attrs['peak']
            self.tsint = hf[name].attrs['tsint']
            self.epstot = hf[name].attrs['epstot']


    def add_figtitle(self, fig):
        fig.suptitle(r'$\tau_s =$ [{:4.2e}, {:4.2e}] $\sigma=${:4.2e} peak={:4.2e} $\epsilon_t = ${:4.2e}'.format(
                     self.tsint[0], self.tsint[1], self.sigma, self.peak, self.epstot));

    def plotAitkenMesh(self, level=-1):
        fig = super().plotAitkenMesh(level=level)
        self.add_figtitle(fig)
        return fig


    def plotFastestMesh(self, level=-1, log=False):
        fig = super().plotFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig


    def plotFastestRelMesh(self, level=-1):
        fig = super().plotFastestRelMesh(level=level)
        self.add_figtitle(fig)
        plt.tight_layout()
        return fig


    def plotAitkenFastestMesh(self, level=-1, log=False):
        fig = plotAitkenFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig


    def plotFastestEigenmode(self, refine=9, level=-1):
        mandex = (np.argmax((-self.fastesteigens[-1,:,:].real).max(axis=1)), np.argmax((-self.fastesteigens[-1,:,:].real).max(axis=0)))
        taus = np.linspace(self.tsint[0],self.tsint[1],2**refine+1)
        ss = streamingtools.StreamingSolver( taus, epstot=self.epstot, beta=self.beta)
        ss.build_system_matrix(self.Kxaxis[mandex[1]], self.Kzaxis[mandex[0]])
        ss.solve_eigen()
        print('Fastest growing eigenmode in new calculation is ',ss.get_fastest_growth())
        fig = plt.figure(figsize=(8,12))
        axs = fig.subplots(nrows=4, ncols=2)
        axs[0][0].set_title('Growing Part')
        axs[0][1].set_title('Oscillating Part')
        for ii,key in enumerate(ss.fastest_eigenvec_by_taus.dtype.names):
            axs[ii][0].plot(ss.taus, ss.fastest_eigenvec_by_taus[key].real,color='C0')
            axs[ii][1].plot(ss.taus, ss.fastest_eigenvec_by_taus[key].imag,color='C1')
            axs[ii][0].set_ylabel(key)
        for ax in axs.flatten():
            ax.set_xlabel(r'$\tau_s$')
            ax.set_xlim((ss.taus.min(),ss.taus.max()))
        fig.suptitle('Dust Fluid Eigenfunctions');
        fig.tight_layout()
        return fig

class PowerBumpGridRun(GridRunBase):

    def __init__(self, name, prefix=''):
        super().__init__(name, prefix=prefix)
        self.name = name

        with h5py.File(os.path.join(prefix,name+'.hdf5'),'r') as hf:
            if 'Kxaxis' in list(hf[name].keys()):
                self.Kxaxis = hf[name+'/Kxaxis'][...]
            if 'Kzaxis' in list(hf[name].keys()):
                self.Kzaxis = hf[name+'/Kzaxis'][...]
            if 'Kxgrid' in list(hf[name].keys()):
                self.Kxgrid = hf[name+'/Kxgrid'][...]
            if 'Kzgrid' in list(hf[name].keys()):
                self.Kzgrid = hf[name+'/Kzgrid'][...]
            if 'aitkengrowthrates' in list(hf[name].keys()):
                self.aitkengrowthrates = hf[name+'/aitkengrowthrates'][...]
            if 'aitkenoscillate' in list(hf[name].keys()):
                self.aitkenoscillate = hf[name+'/aitkenoscillate'][...]
            self.fastesteigens = hf[name+'/fastesteigens'][...]
            if 'backfastesteigens' in list(hf[name].keys()):
                self.backfastesteigens = hf[name+'/backfastesteigens'][...]
            self.beta = hf[name].attrs['beta']
            self.tsint = hf[name].attrs['tsint']
            self.epstot = hf[name].attrs['epstot']
            self.aL = hf[name].attrs['aL']
            self.aP = hf[name].attrs['aP']
            self.bumpfac = hf[name].attrs['bumpfac']
            if 'alleigens' in list(hf[name].keys()):
                self.alleigens = []
                for iz, kz in enumerate(self.Kzaxis):
                    alleigensz = []
                    for ix, kx in enumerate(self.Kxaxis):
                        alleigensx = []
                        for ik,key in enumerate(list(hf[name+'/alleigens/'+str(iz)+'/'+str(ix)].keys())):
                            alleigensx.append(hf[name+'/alleigens/'+str(iz)+'/'+str(ix)+'/'+str(ik)])
                        alleigensz.append(alleigensx)
                    self.alleigens.append(alleigensz)

    def add_figtitle(self,fig):
        fig.suptitle(r'$\tau =$ [{:4.2e}, {:4.2e}] $\epsilon = ${:4.2e} $\beta=${:4.1f} $b_f=${:4.1e} $a_L=${:4.2e} $a_P=${:4.2e}'.format(
                     self.tsint[0], self.tsint[1], self.epstot, self.beta, self.bumpfac, self.aL, self.aP));

    def plotRichConvMesh(self, log=False):
        fig = super().plotRichConvMesh(log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenConvMesh(self, log=False):
        fig = super().plotAitkenConvMesh(log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenMesh(self, level=-1, log=False):
        fig = super().plotAitkenMesh(level=level,log=log)
        self.add_figtitle(fig)
        return fig

    def plotAitkenMeshTol(self, reltol, level=-1, log=False):
        fig = super().plotAitkenMeshTol(reltol, level=level, log=log)
        self.add_figtitle(fig)
        return fig

    def plotFastestMesh(self, level=-1, log=False):
        fig = super().plotFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig

    def plotFastestMeshTol(self, reltol, level=-1, log=False):
        fig = super().plotFastestMeshTol(reltol, level=level)
        self.add_figtitle(fig)
        return fig

    def plotFastestRelMesh(self, level=-1):
        fig = super().plotFastestRelMesh(level=level)
        self.add_figtitle(fig)
        plt.tight_layout()
        return fig


    def plotAitkenFastestMesh(self, level=-1, log=False):
        fig = super().plotAitkenFastestMesh(level=level, log=log)
        self.add_figtitle(fig)
        return fig


    def reSimMode(self, Kx, Kz, ntaus=257, gridding='logarithmic'):
        taus = streamingtools.get_gridding(streamingtools.gridmap[gridding], self.tsint, ntaus)

        ss = streamingtools.StreamingSolverPowerBump(taus, epstot=self.epstot, beta=self.beta,
             aL=self.aL, aP=self.aP, bumpfac=self.bumpfac)
        ss.build_system_matrix(Kx, Kz)
        ss.solve_eigen()
        return ss



class Collector():
    def getGridList(self):
        self.foundfiles = sorted([x for x in os.listdir('.') if re.match('grid[0-9][0-9][0-9].hdf5', x)])
        self.grids = []
        for filename in self.foundfiles:
            grid = GridRun(filename[0:7])
            print('{:s} {:+e} {:+e} {:+e} {:+e} {:d}'.format(grid.name,grid.tsint[0],grid.tsint[1], grid.beta, grid.epstot, grid.fastesteigens.shape[0]))
            self.grids.append({'name':grid.name,
                          'tsint':(grid.tsint[0], grid.tsint[1]),
                          'beta':grid.beta,
                          'epstot':grid.epstot,
                          'refine':grid.fastesteigens.shape[0],
                          'gridObj':grid})

    def findgrid(self,beta=None, tsint=None, epstot=None, refine=None, name=None):
        matches=[]
        for gg in self.grids:
            include = True
            if beta is not None:
                if beta != gg['beta']:
                    include = False
            if tsint is not None:
                if tsint[0] is not None:
                    if tsint[0] != gg['tsint'][0]:
                        include = False
                if tsint[1] is not None:
                    if tsint[1] != gg['tsint'][1]:
                        include = False
            if epstot is not None:
                if epstot != gg['epstot']:
                    include = False
            if refine is not None:
                if refine != gg['refine']:
                    include = False
            if name is not None:
                if name != gg['name']:
                    include = False
            if include:
                matches.append({'name':gg['name'],'gridObj':gg['gridObj']})
        return matches

class LogNormCollector(Collector):
    def getGridList(self):
        self.foundfiles = sorted([x for x in os.listdir('.') if re.match('lognormgrid[0-9][0-9][0-9].hdf5', x)])
        self.grids = []
        for filename in self.foundfiles:
            grid = LogNormGridRun(filename[0:14])
            print('{:s} {:+e} {:+e} {:+e} {:+e} {:+e} {:d}'.format(
                  grid.name,grid.tsint[0],grid.tsint[1], grid.sigma, grid.peak, grid.epstot, grid.fastesteigens.shape[0]))
            self.grids.append({'name':grid.name,
                          'tsint':(grid.tsint[0], grid.tsint[1]),
                          'sigma':grid.sigma,
                          'peak':grid.peak,
                          'epstot':grid.epstot,
                          'refine':grid.fastesteigens.shape[0],
                          'gridObj':grid})

    def findgrid(self,sigma=None, peak=None, tsint=None, epstot=None, refine=None, name=None):
        matches=[]
        for gg in self.grids:
            include = True
            if sigma is not None:
                if sigma != gg['sigma']:
                    include = False
            if peak is not None:
                if beta != gg['peak']:
                    include = False
            if tsint is not None:
                if tsint[0] is not None:
                    if tsint[0] != gg['tsint'][0]:
                        include = False
                if tsint[1] is not None:
                    if tsint[1] != gg['tsint'][1]:
                        include = False
            if epstot is not None:
                if epstot != gg['epstot']:
                    include = False
            if refine is not None:
                if refine != gg['refine']:
                    include = False
            if name is not None:
                if name != gg['name']:
                    include = False
            if include:
                matches.append({'name':gg['name'],'gridObj':gg['gridObj']})
        return matches

class PowerBumpCollector(Collector):
    def getGridList(self):
        self.foundfiles = sorted([x for x in os.listdir('.') if re.match('powerbumpgrid[0-9][0-9][0-9].hdf5', x)])
        self.grids = []
        for filename in self.foundfiles:
            grid = PowerBumpGridRun(filename[0:16])
            print('{:s} {:+e} {:+e} {:+e} {:+e} {:+e} {:+e} {:+e} {:d}'.format(grid.name,grid.tsint[0],grid.tsint[1], grid.beta,
                  grid.epstot, grid.aL, grid.aP, grid.bumpfac,  grid.fastesteigens.shape[0]))
            self.grids.append({'name':grid.name,
                          'tsint':(grid.tsint[0], grid.tsint[1]),
                          'beta':grid.beta,
                          'epstot':grid.epstot,
                          'aL':grid.aL,
                          'aP':grid.aP,
                          'bumpfac':grid.bumpfac,
                          'refine':grid.fastesteigens.shape[0],
                          'gridObj':grid})

    def findgrid(self,beta=None, tsint=None, epstot=None, aL=None, aP=None, bumpfac=None, refine=None, name=None):
        matches=[]
        for gg in self.grids:
            include = True
            if beta is not None:
                if beta != gg['beta']:
                    include = False
            if tsint is not None:
                if tsint[0] is not None:
                    if tsint[0] != gg['tsint'][0]:
                        include = False
                if tsint[1] is not None:
                    if tsint[1] != gg['tsint'][1]:
                        include = False
            if epstot is not None:
                if epstot != gg['epstot']:
                    include = False
            if aL is not None:
                if aL != gg['aL']:
                    include = False
            if aP is not None:
                if aP != gg['aP']:
                    include = False
            if bumpfac is not None:
                if bumpfac != gg['bumpfac']:
                    include = False
            if refine is not None:
                if refine != gg['refine']:
                    include = False
            if name is not None:
                if name != gg['name']:
                    include = False
            if include:
                matches.append({'name':gg['name'],'gridObj':gg['gridObj']})
        return matches



def convcriteria(a,b, reltolimag=0.1, reltolreal=0.05):
    rel = lambda a,b: np.abs(a-b)/np.abs(b)
    relmag = rel(a,b)
    relimag = rel(a.imag,b.imag)
    relreal = rel(a.real,b.real)
    if relimag < reltolimag and relreal < reltolreal:
        return True
    else:
        return False


def richardson(seq):
    """a modification of the routine from mpmath, strips out the oscillating check and uses normal floats"""
    if len(seq) < 3:
        raise ValueError("seq should be of minimum length 3")
    N = len(seq)//2-1
    s = 0.0
    # The general weight is c[k] = (N+k)**N * (-1)**(k+N) / k! / (N-k)!
    # To avoid repeated factorials, we simplify the quotient
    # of successive weights to obtain a recurrence relation
    c = (-1)**N * N**N / scipy.special.factorial(N,exact=True)
    maxc = 1
    for k in range(N+1):
        s += c * seq[N+k]
        maxc = max(abs(c), maxc)
        c *= (k-N)*(k+N+1)**N
        c /= ((1+k)*(k+N)**N)
    return s, maxc

def aitkenize(series):
    """Aitken Delta^2 method applied to input array"""
    aitkenseries = np.zeros(series.shape[0]-2)
    for ia in range(0,series.shape[0]-2):
        pn = series[-3-ia]
        pn1 = series[-2-ia]
        pn2 = series[-1-ia]
        #Abramiwitz & Stegun 3.9.7
        delta2 = pn2 -2.0*pn1 +pn
        aitken = pn - (pn-pn1)**2/delta2
        aitkenseries[series.shape[0]-2-ia-1] = aitken
    return aitkenseries

# This is a utility function for  plotting with pcolormesh
logedges = lambda axis: np.logspace(np.log10(axis[0])- 0.5*(np.log10(axis[1])-np.log10(axis[0])),
                     np.log10(axis[-1])+ 0.5*(np.log10(axis[-1])-np.log10(axis[-2])),num = len(axis)+1)
