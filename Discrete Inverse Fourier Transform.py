import numpy as np, math, random
import matplotlib.pyplot as plt, pylab as pl
import scipy.fftpack
from IPython import display
import time as ttime
import random
from mpl_toolkits.mplot3d import Axes3D
srate = 1000
time = np.arange(0., 2., 1/srate)
pnts = len(time)
signal = 2.5 * np.sin(2 * np.pi * 4 * time) + 1.5 * np.sin(2 * np.pi * 6.5 * time)
fourTime = np.arange(0, pnts)/pnts
fCoefs = np.zeros(len(signal), dtype=complex)
for fi in range(pnts):
	csw = np.exp(-1j*2*np.pi*fi*fourTime)
	# Using - for forward FT
	fCoefs[fi] = np.sum(np.multiply(signal, csw))
ampls = abs(fCoefs)/pnts
# If we apply /pnts for fCoefs in the loop instead of ampls reconSignal would be 0
ampls[1:] = 2 * abs(fCoefs[1:])
hz = np.linspace(0, srate/2, num=int(pnts/2)+1)
plt.stem(hz, ampls[:len(hz)])
# If we don't indicate ampls as ampls[:len(hz)] , Value error would occur regarding the difference in index quantities
plt.xlim([0, 10])
plt.show()
# Using loop
reconSignal = np.zeros(len(signal))
for fi in range(pnts):
	# create coefficient-modulated complex sine wave
	csw = fCoefs[fi] * np.exp(1j*2*np.pi*fi*fourTime)
	# Using +1j for inverse FT
	reconSignal = reconSignal + csw
reconSignal = reconSignal/pnts
# If /pnts is applied in the loop, reconSignal would again appear as zero
# Using fft pack
reconTS = np.real(scipy.fftpack.ifft(fCoefs))*pnts
plt.plot(time, signal, markersize=4, label='original')
plt.plot(time, np.real(reconSignal), 'r.', markersize=1, label='reconstructed')
# We use np.real here because there are some computer rounding errors
plt.plot(time, reconTS, 'g.', markersize=1, label='Reconstructed via  ifft')
plt.legend()
plt.show()
# Bandpass filtering
# srate = 1000
time = np.arange(0, 2, 1/srate)
pnts = len(time)
signal = np.sin(2*np.pi*4*time) + np.sin(2*np.pi*10*time)
fourTime = np.array(np.arange(0, pnts))/pnts
fCoefs   = np.zeros(len(signal), dtype=complex)
for fi in range(pnts):
	csw = np.exp(-1j*2*np.pi*fi*fourTime)
	fCoefs[fi] = np.sum(np.multiply(signal, csw))/pnts
hz = np.linspace(0, srate/2, num=int(pnts/2)+1)
# Finding the coefficient for 10 hz
freqidx = np.argmin(np.abs(hz-10))
fCoefsMod = list(fCoefs)
# If we don't use list fcoef would change into fcoefMod
fCoefsMod[freqidx] = 0
# This is only for positive frequency. if we plot this , it wouldn't be accurate
fCoefsMod[pnts-freqidx] = 0
# fCoefsMod[-freqidx] = 0 For negative frequency
reconMod = np.zeros(len(signal))
for fi in range(pnts):
	csw = fCoefsMod[fi] * np.exp(1j*2*np.pi*fi*fourTime)
	reconMod = reconMod + csw
# There are normalizations for the forward and inverse Fourier transforms
# they cancel each other out when done in succession. You don't apply normalization.
plt.plot(time, signal)
plt.title('Original signal, time domain')
plt.show()
plt.stem(hz, 2*np.abs(fCoefs[0:len(hz)]))
plt.xlim([0, 15])
plt.title('Original signal, frequency domain')
plt.show()
plt.plot(time, np.real(reconMod))
plt.title('Band-stop filtered signal, time domain')
plt.show()
