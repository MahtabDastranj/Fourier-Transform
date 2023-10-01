import math, random, timeit
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from mpl_toolkits.mplot3d import axes3d
# Sampling rate
srate = 1000
pnts = 1000
signal = np.random.randn(pnts)
hz = np.linspace(0, srate/2, int(pnts/2)+1)
freqs = np.mean(np.diff(hz))        # Or srate/pnts
# WE can enhance frequency resolution by increasing pnts
srate = 10
time = np.arange(0, 10, 1/srate)
signal[0: int(len(time) * .1)] = 1
signalX = scipy.fftpack.fft(signal)
frequnits = np.linspace(0, srate, len(time))
# It goes up to 2*Nyquist there isn't actually frequencies we demonstrated.There could be negative frequencies displayed
plt.plot(time, signal[:len(time)], 's-')
# Using stem plot would be better since only plotting it would imply quantities for amounts we don't know
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.show()
plt.plot(hz, 2*abs(signalX[:len(hz)]), 's-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency domain')
plt.show()
## Time domain zero padding
signal = np.hanning(40)     # Half a cosine wave
signalX = scipy.fftpack.fft(signal)/len(signal)
ampl = 2 * np.abs(signal)
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(signal, 'ks-')
plt.xlim(0, 80)
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.title('Time domain')
axs[1, 0].plot(frequnits, ampl[:len(frequnits)])
plt.xlim(-.01, .3)
plt.xlabel('Frequency (a.u.)')
plt.ylabel('Amplitude')
plt.title('Frequency domain')
signal = np.concatenate((signal, np.zeros(len(signal))), axis=0)
axs[0, 1].plot(signal, 'ks-')
plt.xlim(0, 80)
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.title('Time domain')
axs[1, 1].plot(frequnits, ampl)
plt.xlim(-.01, .3)
plt.xlabel('Frequency (a.u.)')
plt.ylabel('Amplitude')
plt.title('Frequency domain')
plt.tight_layout()
plt.show()
