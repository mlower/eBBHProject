'''
paul.lasky@monash.edu

tools for calculating gravitational-wave memory waveforms
'''


import numpy as np

import QNMs
import tools

# QNMs = reload(QNMs)
# tools = reload(tools)

def MWM(time, m1, m2, dist, Theta, psi, rm = 3.):
    '''
    Calculates the minimal waveform model memory waveform: eqns (5) and (9) from Favata (2009. ApJL 159)   

    time: time array in seconds
    m1, m2: component masses in solar units
    dist: distance to source in Mpc
    Theta: inclination angle
    psi: polarisation angle   
    '''

    # convert everything to geometric units
    time_geo = tools.time_s_to_geo(time)    # units: metres
    
    m1_geo = tools.m_sol_to_geo(m1)         # units: metres
    m2_geo = tools.m_sol_to_geo(m2)         # units: metres

    dist_geo = tools.dist_Mpc_to_geo(dist)  # units: metres

    # total mass
    MM = m1_geo + m2_geo

    # symmetric mass ratio
    eta = tools.m12_to_symratio(m1_geo, m2_geo)

    ## this is the orbital separation at the matching radius -- see Favata (2009) before eqn (8).
    # the default value for this is given as rm = 3 MM.
    rm *= MM
    
    # calculate dimensionless mass and spin of the final black hole
    # from the Buonanno et al. (2007) fits
    Mf_geo, jj = QNMs.final_mass_spin(m1_geo, m2_geo)
    
    # calculate the QNM frequencies and damping times
    # from the fits in Table VIII of Berti et al. (2006)
    omega220, tau220 = QNMs.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=0)
    omega221, tau221 = QNMs.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=1)
    omega222, tau222 = QNMs.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=2)

    sigma220 = 1j * omega220 + 1./tau220
    sigma221 = 1j * omega221 + 1./tau221
    sigma222 = 1j * omega222 + 1./tau222
    
    # set the time at merger to be at t = 0
    tm = 0.
    TT = time_geo - tm
    
    # some quantity defined after equation (7) of Favata
    trr = 5. * MM * rm**4 / (256. * eta * MM**4)

    ## calculate the A_{ell m n} matching coefficients.  Note that I've solved
    ## a matrix equation that solved for the three coefficients fron three equations
    xi = 2. * np.sqrt(2. * np.pi / 5.) * eta * MM * rm**2
    chi = -2. * 1j * np.sqrt(MM / rm**3)

    A220 = xi * (sigma221 * sigma222 * chi**2 + sigma221 * chi**3 + sigma222 * chi**3 + chi**4) / ((sigma220 - sigma221) * (sigma220 - sigma222))
    A221 = xi * (sigma220 * sigma222 * chi**2 + sigma220 * chi**3 + sigma222 * chi**3 + chi**4) / ((sigma221 - sigma220) * (sigma221 - sigma222))
    A222 = xi * (sigma220 * sigma222 * chi**2 + sigma220 * chi**3 + sigma221 * chi**3 + chi**4) / ((sigma221 - sigma222) * (sigma220 - sigma222))

    ## Calculate the coefficients in the summed term of equation (9) from Favata (2009)
    # this is a double sum, with each variable going from n = 0 to 2; therefore 9 terms
    coeffSum00 = sigma220 * np.conj(sigma220) * A220 * np.conj(A220) / (sigma220 + np.conj(sigma220))
    coeffSum01 = sigma220 * np.conj(sigma221) * A220 * np.conj(A221) / (sigma220 + np.conj(sigma221))
    coeffSum02 = sigma220 * np.conj(sigma222) * A220 * np.conj(A222) / (sigma220 + np.conj(sigma222))

    coeffSum10 = sigma221 * np.conj(sigma220) * A221 * np.conj(A220) / (sigma221 + np.conj(sigma220))
    coeffSum11 = sigma221 * np.conj(sigma221) * A221 * np.conj(A221) / (sigma221 + np.conj(sigma221))
    coeffSum12 = sigma221 * np.conj(sigma222) * A221 * np.conj(A222) / (sigma221 + np.conj(sigma222))

    coeffSum20 = sigma222 * np.conj(sigma220) * A222 * np.conj(A220) / (sigma222 + np.conj(sigma220))
    coeffSum21 = sigma222 * np.conj(sigma221) * A222 * np.conj(A221) / (sigma222 + np.conj(sigma221))
    coeffSum22 = sigma222 * np.conj(sigma222) * A222 * np.conj(A222) / (sigma222 + np.conj(sigma222))
    
    ## radial separation
    rr = rm * (1. - TT / trr)**(1./4.)

    ## set up strain
    h_MWM = np.zeros(len(TT))

    ## calculate strain for TT < 0.
    h_MWM[TT <= 0.] = 8. * np.pi * MM / rr[TT <= 0.]

    ## calculate strain for TT > 0.
    term00 = coeffSum00 * (1. - np.exp(-TT[TT > 0.] * (sigma220 + np.conj(sigma220))))
    term01 = coeffSum01 * (1. - np.exp(-TT[TT > 0.] * (sigma220 + np.conj(sigma221))))
    term02 = coeffSum02 * (1. - np.exp(-TT[TT > 0.] * (sigma220 + np.conj(sigma222))))

    term10 = coeffSum10 * (1. - np.exp(-TT[TT > 0.] * (sigma221 + np.conj(sigma220))))
    term11 = coeffSum11 * (1. - np.exp(-TT[TT > 0.] * (sigma221 + np.conj(sigma221))))
    term12 = coeffSum12 * (1. - np.exp(-TT[TT > 0.] * (sigma221 + np.conj(sigma222))))

    term20 = coeffSum20 * (1. - np.exp(-TT[TT > 0.] * (sigma222 + np.conj(sigma220))))
    term21 = coeffSum21 * (1. - np.exp(-TT[TT > 0.] * (sigma222 + np.conj(sigma221))))
    term22 = coeffSum22 * (1. - np.exp(-TT[TT > 0.] * (sigma222 + np.conj(sigma222))))

    sum_terms = np.real(term00 + term01 + term02 +
                        term10 + term11 + term12 +
                        term20 + term21 + term22)

    h_MWM[TT > 0.] = 8. * np.pi * MM / rm + sum_terms / (eta * MM)

    ## calculate the plus polarisation of GWs: eqn. (5) from Favata (2009)
    sT = np.sin(Theta)
    cT = np.cos(Theta)

    h_plus_coeff = 0.77 * eta * MM / (384. * np.pi) * sT**2 * (17. + cT**2) / dist_geo
    hp_nonrot = h_plus_coeff * h_MWM

    ## rotate according to the polarisation angle, psi
    hp = np.cos(2. * psi) * hp_nonrot;
    hx = np.sin(2. * psi) * hp_nonrot;
    
    return hp, hx 
