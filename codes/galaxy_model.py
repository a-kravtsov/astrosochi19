#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  a template for a simple galaxy formation model a la Krumholz & Dekel (2012); 
#                    see also Feldmann 2013
#
#   Andrey Kravtsov, 2019
#
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from colossus.cosmology import cosmology
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp2d

   
class simplest_model(object):

    def __init__(self,  t = None, Mh = None, Mg = None, Ms = None, verbose = False, **kwargs):
        cosmo = kwargs['cosmo']
        sfrmodel = kwargs['sfrmodel']
        tausf = kwargs['tausf'] 
        if cosmo is not None: 
            self.cosmo = cosmo
            self.fbuni = cosmo.Ob0/cosmo.Om0
        else:
            errmsg = 'to initialize gal object it is mandatory to supply the collossus cosmo(logy) object!'
            raise Exception(errmsg)
            return
            
        if Mh is not None: 
            self.Mh = Mh
        else:
            errmsg = 'to initialize gal object it is mandatory to supply Mh!'
            raise Exception(errmsg)
            return
            
        if t is not None: 
            self.t = t # in Gyrs
            self.z = self.cosmo.age(t, inverse=True)        
            self.gr = self.cosmo.growthFactor(self.z)
            self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
            self.thubble = self.cosmo.hubbleTime(self.z)
            self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        else:
            errmsg = 'to initialize gal object it is mandatory to supply t!'
            raise Exception(errmsg)
            return
            
        
        if Ms is not None:
            self.Ms = Ms
        else: 
            self.Ms = 0.0
        if Mg is not None:
            self.Mg = Mg
        else: 
            self.Mg = self.fbuni*Mh
            
        
        # only one simplest model based on total gas density and constant tau_sf 
        sfr_models = getattr(self, 'sfr_models', None)
        if sfr_models is None:
            self.sfr_models = {'gaslinear': self.SFRgaslinear}
    
        if not hasattr(self, 'sfrmodel'):
            try: 
                self.sfr_models[sfrmodel]
            except KeyError:
                print("unrecognized sfrmodel in model_galaxy.__init__:", sfrmodel)
                print("available models:", self.sfr_models)
                return
            self.sfrmodel = sfrmodel
        else:
            if self.sfrmodel is None:
                errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
                raise Exception(errmsg)
            return
            
        if self.sfrmodel is 'gaslinear':
            self.tausf = tausf
            
        if verbose is not None:
            self.verbose = verbose
        else:
            errmsg = 'verbose parameter is not initialized'
            raise Exception(errmsg)
            return

        self.Mgin = self.Mg_in(t); 
        self.sfr = self.SFR(t)
        self.Rloss1 = 1.0 - self.R_loss(t)

        return
        
    
    def dMhdt(self, Mcurrent, t):
        """
        halo mass accretion rate using approximation of eqs 3-4 of Krumholz & Dekel 2012
        output: total mass accretion rate in Msun /Gyr
        """
        self.Mh = Mcurrent
        tau_acc = 0.175 * t**1.8 * (Mcurrent/1.e12)**(-0.1)
        mdot = Mcurrent / tau_acc
        return mdot

    def Mg_in(self, t):
        dummy = self.fbuni*self.fg_in(t)*self.dMhdt(self.Mh,t)
        return dummy
        
    def fg_in(self,t):
        return 1.0

    def tau_sf(self):
        """
        gas consumption time in Gyrs 
        """
        return self.tausf

    def SFRgaslinear(self, t):
        return self.Mg/self.tau_sf()
         
    def SFR(self, t):
        """
        master routine for SFR - 
        eventually can realize more star formation models
        """  
        return self.sfr_models[self.sfrmodel](t)
        
    def dMsdt(self, Mcurrent, t):
        dummy = self.Rloss1*self.sfr
        return dummy
 
    def R_loss(self, t):
        """
        fraction of mass formed in stars that is returned back to the ISM
        """
        return 0.46
        
    def dMgdt(self, Mcurrent, t):
        dummy = self.Mgin - self.Rloss1 * self.sfr 
        return dummy
        
    def evolve(self, Mcurrent, t):
        # first set auxiliary quantities and current masses
        self.z = self.cosmo.age(t, inverse=True)        
        self.gr = self.cosmo.growthFactor(self.z)
        self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
        self.thubble = self.cosmo.hubbleTime(self.z)
        self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        self.Mh = Mcurrent[0]; self.Mg = Mcurrent[1]; 
        self.Ms = Mcurrent[2]; 
        self.Mgin = self.Mg_in(t); 
        self.Rloss1 = 1.0 - self.R_loss(t)
        self.sfr = self.SFR(t)
        
        # calculate rates for halo mass, gas mass, stellar mass, and mass of metals
        dMhdtd = self.dMhdt(Mcurrent[0], t)
        dMgdtd = self.dMgdt(Mcurrent[1], t)
        dMsdtd = self.dMsdt(Mcurrent[2], t)
        
        if self.verbose:
            print("evolution: t=%2.3f Mh=%.2e, Mg=%.2e, Ms=%.2e, SFR=%4.2e"%(t,self.Mh,self.Mg,self.Ms,self.SFR(t)*1.e-9))

        return [dMhdtd, dMgdtd, dMsdtd]
        
    def test_galaxy_evolution(gmodel=None, Minit=None, t_output=None, kwargs=None):
        """
        a simple helper routine to generate a single evolutionary track for an 
        object of initial mass Minit (in Msun) at redshift zmax. The track will be
        generated from zmax to zmin at nzb equally spaced redshifts. 
        
        Parameters:
        -----------
            Minit: float, initial total mass at zmax
            zmax, zmin: floats, maximum and minimum redshifts of the evolutionary track
            nzb: integer, number of epochs at which to output variables
            sfrmodel: string, model of star formation to use; only 'gaslinear'; in this simplest model 
            tausf: float, constant gas depletion time scale
            cosmo: colossus cosmology object
            verbose: logical, default=False, whether to print auxiliary information during run
        
        Returns:    
        --------
            zg, t_output: numpy float arrays containing grids of output redshifts and times
            Mout: numpy 2d array, the first axis contains time evolution, while 2nd particular variable
                Mout[:,0] - evolution of total halo mass
                Mout[:,1] - evolution of ISM gas mass
                Mout[:,2] - evolution of stellar mass
        """

        # instantiate a galaxy model at t=tinit assuming a given star fromation model and cosmology
        g = gmodel(t = t_output[0], Mh = Minit, Mg = None, Ms = None, **kwargs)
    
        # array of quantities to evolve with ODE solver
        y0 = np.array([g.Mh, g.Mg, g.Ms])
        # evolve y0 in time
        Mout = odeint(g.evolve, y0, t_output, rtol = 1.e-7, mxstep = 4000)
    
        Mout = np.clip(Mout[:,:], 0.0, 1.e100)

        return t_output, Mout
        
    def plot_test_evolution(Mout, t_output=None, gmodel=None, savefig=None, kwargs=None):
        cosmo = kwargs['cosmo']
        tu = cosmo.age(0.0) # the age of the Universe
        fig, ax = plt.subplots(1,2, figsize=(6,3))    
        ax[0].set_xlabel(r'$t\ \rm (Gyr)$'); ax[0].set_ylabel(r'$M_{\rm h},\ M_{\rm g},\ M_{*}\ (\rm M_\odot)$')
        ax[0].set_xlim(0.,tu); ax[0].set_ylim(1.e4,1.e13)
        ax[0].set_yscale('log')
        ax[0].plot(t_output, Mout[:,0], lw = 3.0, label=r'$M_{\rm h}$')
        ax[0].plot(t_output, Mout[:,1], lw = 3.0, label=r'$M_{\rm g, ISM}$')
        ax[0].plot(t_output, Mout[:,2], lw = 3.0, label=r'$M_{\star}$')

        ax[0].legend(frameon=False,loc='lower right', ncol=2, fontsize=9)
        ax[0].grid(linestyle='dotted', c='lightgray')
    
        # plot SFR(t)
        ax[1].set_xlabel(r'$t\ \rm (Gyr)$'); ax[1].set_ylabel(r'$\rm\ SFR\ (M_\odot/yr)$')
        ax[1].set_xlim(0.,tu); ax[1].set_ylim(1.e-4,50.)
        ax[1].set_yscale('log')
        ax[1].yaxis.set_label_position('right')

        SFR = np.zeros_like(t_output); 

        for i, td in enumerate(t_output):
            ge = gmodel(t = td, Mh = Mout[i,0], Mg = Mout[i,1], Ms = Mout[i,2],
                                    **kwargs)
            SFR[i] = ge.SFR(td)*1.e-9;

        ax[1].plot(t_output, SFR, lw = 3.0, label=r'$\rm SFR$')
        ax[1].legend(frameon=False,loc='lower right', ncol=2, fontsize=9)
        ax[1].grid(linestyle='dotted', c='lightgray')
        if savefig != None: 
            plt.savefig(savefig, bbox_inches='tight')
        plt.show()
        return 

    def compute_mass_grid(gmodel=None, lMhig=None, t_output=None, verbose=False, kwargs=None):
        Nt = np.size(t_output)
        Nm = np.size(lMhig) # mass grid size
        Mh  = np.power(10., lMhig)
        cosmo = kwargs['cosmo']
        zg = cosmo.age(t_output, inverse=True)
        lzg1 = np.log10(1.0+zg)
 
        # arrays to hold evolutionary grids of halo mass, gas mass, stellar mass and metal mass
        Mhout = np.zeros((Nm,Nt)); Mgout = np.zeros((Nm,Nt))
        Msout = np.zeros((Nm,Nt)); 
        #evolve a grid of halo masses
        print("evolving mass grid...")
        for j, Mhd in enumerate(Mh):
            g = gmodel(t = t_output[0], Mh = Mhd, Mg = None, Ms = None, 
                        verbose = verbose, **kwargs)
            # initial values of masses
            y0 = np.array([g.Mh, g.Mg, g.Ms])
            # solve the system of ODEs
            Mout = odeint(g.evolve, y0, t_output, rtol = 1.e-5, mxstep = 4000)
            # split Mout into arrays for specific masses with more intuitive names for convenience
            Mhout[j,:] = Mout[:,0]; Mgout[j,:] = Mout[:,1]
            Msout[j,:] = Mout[:,2];
        print("done.")

        # 
        # prepare 2D splines for interpolation
        #
        Mmin = 1.e-10 # prevent zeros in case no SF occurred
        lMhi = interp2d(lzg1, lMhig, np.log10(np.maximum(Mmin,Mhout)), bounds_error=True, kind='cubic')
        lMgi = interp2d(lzg1, lMhig, np.log10(np.maximum(Mmin,Mgout)), bounds_error=True, kind='cubic')
        lMsi = interp2d(lzg1, lMhig, np.log10(np.maximum(Mmin,Msout)), bounds_error=True, kind='cubic')

        return lMhi, lMgi, lMsi
     

        
