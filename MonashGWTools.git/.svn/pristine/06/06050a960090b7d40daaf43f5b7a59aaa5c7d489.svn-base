

'''
Paul Lasky
Handy tools
'''

import numpy as np
from scipy.interpolate import interp1d
import constants as cc

def m12_to_mc(m1, m2):
    # convert m1 and m2 to chirp mass
    return (m1*m2)**(3./5.) / (m1 + m2)**(1./5.)

def m12_to_symratio(m1, m2):
    # convert m1 and m2 to symmetric mass ratio
    return m1 * m2 / (m1 + m2)**2


def m_sol_to_geo(mm):
    # convert from solar masses to geometric units
    return mm / cc.kg * cc.GG / cc.cc**2

def m_geo_to_sol(mm):
    # convert from geometric units to solar masses
    return mm * cc.kg / cc.GG * cc.cc**2

def time_s_to_geo(time):
    # convert time from seconds to geometric units
    return time * cc.cc

def time_geo_to_s(time):
    # convert time from seconds to geometric units
    return time / cc.cc

def freq_Hz_to_geo(freq):
    # convert freq from Hz to geometric units
    return freq / cc.cc

def freq_geo_to_Hz(freq):
    # convert freq from geometric units to Hz
    return freq * cc.cc

def dist_Mpc_to_geo(dist):
    # convert distance from Mpc to geometric units (i.e., metres)
    return dist * cc.Mpc

def h_tot(hp, Fp, hx, Fx):
    # calculate h_total from plus and cross polarizations and antenna pattern
    return Fp * hp + Fx * hx


def nfft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)
    
    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs
    
    return hf, ff

def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h


def inner_product(aa, bb, freq, PSD):
    '''
    Calculate the inner product defined in the matched filter statistic
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    PSD_interp = PSD_interp_func(freq)

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = freq[1] - freq[0]
    integral = np.sum(integrand) * df

    product = 4. * np.real(integral)

    return product


def snr_exp(aa, freq, PSD):
    '''
    Calculates the expectation value for the optimal matched filter SNR

    snr_exp(aa, freq, PSD)
    '''
    return np.sqrt(inner_product(aa, aa, freq, PSD))

    



