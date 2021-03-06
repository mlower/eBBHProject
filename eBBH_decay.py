#! /usr/bin/env/python3

# This program solves the coupled set of ODEs given by equations 5.6 & 5.7 in Peters (1964)
# Marcus Lower (2017)

import numpy as np

import astropy.constants as c
import astropy.units as u

from scipy import integrate

import matplotlib.pyplot as plt

## Binary attributes:

m1 = 9.1*u.Msun
m2 = 8.2*u.Msun

Rs1 = ((2*c.G*m1)/(c.c**2)).si.value
Rs2 = ((2*c.G*m2)/(c.c**2)).si.value

P = 1.*u.day

## Initial conditions

ai = ((c.G *(m1 + m2)*P**2/(4.*np.pi**2))**(1./3.)).si.value
ei = 0.6

## Solving Eqns 5.6 - 5.7 from Peters (1964):

def dOdt(O,t=0):
    '''
    O = [semi-major axis, eccentricity]
    '''
    alpha = (c.G**3*m1*m2*(m1+m2)/c.c**5).si
    return np.array([
        (-64./5. * alpha/(O[0]**3*(1. - O[1]**2)**(7./2.)) * (1. + 73./24.*O[1]**2 + 37./96.*O[1]**4)).si.value,
        (-304./15. * O[1] * alpha/(O[0]**4*(1. - O[1]**2)**(5./2.)) * (1. + 121./304.*O[1]**2)).si.value
    ])

tmax = (1.5e9*u.yr).si.value                 # maximum time value
time = np.logspace(-1, np.log10(tmax), 3000) # array of 3000 time values
Init = np.array([ai, ei])                    # initial conditions

O, infodict = integrate.odeint(dOdt, Init, time, full_output=True)

infodict['message']

## Calculating orbital period and GW frequency:

Period = ((O[:,0]**2 * (4.*np.pi**2)/(c.G*(m1+m2)))**(1/2)).si.value
GWfreq = 2/(Period)

## Plotting semi-major axis vs time & eccentricity vs time:

time_year = (time * u.s).to('yr').value

f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.loglog(time_year,O[:,0],color='red')
plt.title('eBBH semi-major axis decay')
plt.xlabel('Time [years]')
plt.ylabel(r'Semi-Major Axis [m]')
plt.xlim(1e7,max(time_year))
plt.ylim(1e8,1e10)
plt.savefig('SemiMajAxis_decay.pdf')

f2 = plt.figure(2)
ax2 = f2.add_subplot(111)
ax2.semilogx(time_year,O[:,1],color='blue')
plt.title('eBBH eccentricity decay')
plt.xlabel('Time [years]')
plt.ylabel(r'Eccentricity')
plt.xlim(1e7,max(time_year))
plt.ylim(1e-7,1)
plt.savefig('Eccen_decay.pdf')

f3 = plt.figure(3)
ax3 = f3.add_subplot(111)
ax3.loglog(GWfreq,O[:,1],color='blue')
plt.title('eBBH eccentricity vs GW frequency')
plt.xlabel('GW frequency [Hz]')
plt.ylabel(r'Eccentricity')
plt.xlim(2,4000)
plt.ylim(1e-7,1)
plt.savefig('EccenVsGWfreq.pdf')

f4 = plt.figure()
ax4 = f4.add_subplot(111)
ax4.semilogx(O[:,0],O[:,1],'o',color='darkorange')
plt.title('eBBH eccentricity vs semi-major axis')
plt.xlabel('Semi-major axis [m]')
plt.ylabel(r'Eccentricity')
plt.xlim(1e4,1e10)
plt.ylim(1e-8,1)
plt.savefig('EccenVsSemiMajorAxis.pdf')

plt.show(1); plt.show(2); plt.show(3); plt.show(4)
