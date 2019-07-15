import numpy as np
from colossus.cosmology import cosmology
from colossus.halo.mass_so import M_to_R
from colossus.lss import peaks
from scipy.optimize import fsolve

# year in seconds
yr = 365.*24.*3600.
# Mpc in km
Mpc2km = 3.0865e19 

def eta_func(x, toB): 
    """
    auxiliary function to solve equation for eta numerically
    """
    return x - np.sin(x) - toB

def AB(Ri, Mi, Vi):
    """
    Function to compute the coefficients for the r and t solutions 
    and initial total energy of the perturbation. 
    
    Parameters: 
    -----------    
        Ri: float in physical Mpc; 
        Mi: float in Msun, 
        Vi: float in km/s
        
    Returns:
    --------
        A, B: floats, constants forthe  parametric solutions
              for r and t in Mpc and Gyr, respectively
    """
    GMi = 4.30131e-9 * Mi
    Ei = 0.5*np.square(Vi) - GMi/Ri
    Ei2 = 2.0*np.abs(Ei)
    A = GMi / Ei2
    B = 4.20588e-6 * Mi / Ei2**(1.5) # B in Gyrs
    return A, B, Ei
    
def solve_spherical_collapse_for_M(zi, zg, M, delta_i, cosmo_model):
    """
    setup initial conditions and construct a solution of spherical collapse
    model for an initial perturbation with specified mass M, initial
    overdensity delta_i at z_i for a cosmological model specified in the
    dictionary cosmo_model (to be passed to the colossus routines).
    
    WARNING! This routine is currently set up to produce solutions 
    ONLY FOR COSMOLOGIES WITHOUT DARK ENERGY (Omega_Lambda=0)
    
    Parameters:
    ----------- 
        zi: float
                initial redshift 
        zg: array of floats
                grid of redshifts for which to compute solution
        M: float
                mass of the perturbation in /h Msun
        delta_i: float
                initial overdensity
        cosmo_model: colossus cosmology dictionary 
                with the desired cosmology
                
    Returns:
    --------
    R:  float numpy vector of physical radii of the perturbation at different t
    t:  float numpy vector: the times at which solution is given
    RH: float numpy vector, expansion factor of the universe normalized to 
        the initial physical radius of the perturbation,
    delta: float numpy vector: overdensities of spherical collapse solution at t
    """
    cosmo = cosmology.setCosmology('runcosmo', cosmo_model )

    ag = 1.0/(1.+zg)
    ai = 1.0/(1.+zi)
    ti = cosmo.age(zi)
    Omz = cosmo.Om(zi) # Omega_m(zi)
    
    R_L = peaks.lagrangianR(M) 
    Mi = M / cosmo.h
    # corresponding peak height
    nu_i = 1.69 / cosmo.sigma(R_L, z=0, j=0)
    # Lagrangian radius of the perturbation
    # Actual initial physical radius of the perturbation in Mpc
    Rip = R_L * ai / (1.0 + delta_i)**(1./3.) / cosmo.h 
    
    #DDr = growth_factor_derivative()
    # dDdt = dD_+/dt = dD_+/dz*dz/dt
    dDdt = cosmo.growthFactor(zi, derivative=1) * cosmo.age(ti, derivative=1, inverse=True)
    # DDr = dD_+(ti)/dt/ D_+(ti)
    DDr = dDdt/cosmo.growthFactor(zi)/(1.e9*yr)
    #DDr2 = cosmo.Hz(zi) * 3.24078e-20 
    # initial velocity
    VHi = cosmo.Hz(zi) * Rip 
    Vi = VHi - Rip * Mpc2km/3.*delta_i/(1.+delta_i)*DDr
    delta_lin0 = delta_i / cosmo.growthFactor(zi)

    if (Omz < 1.0) and (delta_i < 1./Omz-1.):
        print("error: perturbation with specified parameters will not collapse")
        print("error: delta_i must be smaller than 1/Omega(zi)-1 for open cosmologies")
        return
    
    if delta_lin0 < 1.69:
        print("warning: delta_lin(z=0)=%.3f and perturbation will not collapse by z=0"%delta_lin0)

    A, B, Ei = AB(Rip, Mi, Vi)
    tcoll = B * 2.0 * np.pi 

    tuni = cosmo.age(zg)
    eta = np.zeros_like(tuni)
    # for each time solve for corresponding eta
    for i, td in enumerate(tuni):
        toB = td/B 
        eta[i] = fsolve(eta_func, x0=1., args=(toB))[0]

    # compute solution for perturbation evolution in the parametric form
    t = B * (eta - np.sin(eta)) + ti
    R = A * (1.0 - np.cos(eta))
    # expansion factor normalized to the size of the pertubation at zi
    RH = Rip * (ag/ai)
    # mean density of the universe at each z of the grid
    rhom = cosmo.rho_m(zg)*cosmo.h**2 # h^2 Msun/kpc^3 -> Msun/Mpc^3
    # density within collapsing perturbation
    rho = 3.0 * Mi/(4.*np.pi*(np.clip(R,1.e-7,1.e30))**3) * 1.e-9
    # overdensity of the perturbation at redshifts in zg
    delta =  rho/rhom - 1.0
    
    return R, t, RH, delta, tcoll
