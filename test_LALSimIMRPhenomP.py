#!/bin/python -u
# demonstration script to generate time series of CBC signals
# before running: jhbuild shell
# in order to run: ipython cbc_timeseries.py
# eric.thrane@ligo.org

## imports
import numpy as np
# the modules directory is from Pablo's tbs_scripts/python/tbs_sim
import modules.common as cm
from modules.gaussian_noise import gaussian_noise
from modules.generateIMRPhenomPv2 import FD
from modules.time_estimate import time_estimate
# standard imports
import os, sys, importlib
# add plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as py
# debugging
import pdb
# import specialty functions for lalproject
from modules.calbasis import calbasis
# import Paul's GW tools for inner products
import spectroscopy.gwtools_pdl.tools as gwt

# parameters
fmin=10;  fmax=1024
# frequency spacing (must be 2^n), reference frequency
# deltaF is chosen to be small enough so that the duration is long enough
# so that there is no wrap-around! Do not change deltaF without checking this
deltaF=1.0/32;  fRef=100
S1=[0, 0, 0];  S2=[0, 0, 0]
ras=0;  dec=0
inc=0;  pol=0;  phi=0
tc = 1164345910
d=410; m1=20; m2=20

# generate waveform in the frequency domain (single detector = H1) from
# posterior samples
hpsf,f = FD(fmin, fmax, deltaF, d, m1, m2, S1, S2, fRef, inc, ras, dec, pol, phi, tc, 'H1')

# print first five elements of FFT for diagnostic
print "  ",;  print " ".join([str(x) for x in hpsf[1:5]] )

# time series
fs=2*fmax;  dt=1.0/fs
dur = 1/deltaF
t = np.linspace(0, dur, dur/dt)
#hpst = gwt.ifft_eht(hpsf, fs)
hpst = np.fft.irfft(hpsf)*fs

# diagnostic plot
py.figure()
#pdb.set_trace()
py.plot(t, hpst, 'g')
py.xlabel('t (s)')
py.ylabel('strain')
py.savefig('img/test_LALSimIMRPhenomP.png')

# leave for loop with indents
print "done"
