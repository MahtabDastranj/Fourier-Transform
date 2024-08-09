import timeit
<<<<<<< Updated upstream
import numpy as np, math, random
import numpy.matlib
=======

import numpy as np, math, random
>>>>>>> Stashed changes
import matplotlib.pyplot as plt, pylab as pl
import scipy.fftpack
from IPython import display
import time as ttime
import random
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< Updated upstream
pnts = 1000
signal = np.random.randn(pnts)
# timer for "slow" Fourier transform
tic = timeit.default_timer()
fourTime = np.array(pnts)/pnts
fCoefs = np.zeros(len(signal), dtype=complex)
for fi in range(pnts):
    csw = np.exp(-1j*2*np.pi*fi*fourTime)
    fCoefs[fi] = np.sum(np.multiply(signal, csw))/pnts
toc = timeit.default_timer()
t1 = toc - tic
# time for "fast" Fourier transform
tic = timeit.default_timer()
# FFT needs normalization for fcoefs and ampls
fCoefsF = scipy.fftpack.fft(signal)/pnts
toc = timeit.default_timer()
t2 = toc - tic
plt.bar([1, 2], [t1, t2])
plt.title('Computation times')
plt.ylabel('Time (sec.)')
plt.xticks([1, 2], ['loop', 'FFT'])
plt.show()
# IFFT
srate = 1000
time = np.arange(0, 3, 1/srate)
pnts = len(time)
# create multi spectral signal
signal = np.multiply((1+np.sin(2*np.pi*12*time)), np.cos(np.sin(2*np.pi*25*time)+time))
signalX = scipy.fftpack.fft(signal)
reconSig = scipy.fftpack.ifft(signalX)
plt.plot(time, signal, label='Original')
plt.plot(time, np.real(reconSig), label='Reconstructed')
plt.xlabel('Time (sec.)')
plt.ylabel('amplitude (a.u.)')
plt.show()
# The perfection of the fourier transform
fourTime = np.array(range(pnts))/pnts
F = np.zeros((pnts, pnts), dtype=complex)
for fi in range(pnts):
    csw = np.exp(-1j*2*np.pi*fi*fourTime)
    # put csw into column of matrix F
    F[:, fi] = csw
# compute inverse of F (and normalize by N)
Finv = np.linalg.inv(F)*pnts
fig, axs = plt.subplots(2)      # demonstrated vertically
axs[0].plot(fourTime, np.real(F[:, 5]), label='real')
axs[0].plot(fourTime, np.imag(F[:, 5]), label='imag')
plt.title('One column of matrix F')
plt.legend()
axs[1].plot(fourTime, np.real(Finv[:, 5]), label='real')
axs[1].plot(fourTime, np.imag(Finv[:, 5]), label='imag')
# We can divide each signal to csw with little frequencies
plt.title('One column of matrix F^${-1}$')
plt.legend()
plt.tight_layout()
plt.show()
plt.figimage(np.real(F))
plt.show()
srate = 400
time = np.arange(0, srate*2, 1/srate)
pnts = len(time)
nreps = 50
# matlib.repmat(a, m, n) Repeats a 0-D to 2-D array or matrix MxN times
data = np.matlib.repmat(np.sin(2*np.pi*10*time), nreps, 1)
dataX1 = scipy.fftpack.fft(data, axis=0)/pnts
dataX2 = scipy.fftpack.fft(data, axis=1)/pnts
hz = np.linspace(0, pnts/2, int(pnts/2)+1)
print(np.shape(dataX1), np.shape(dataX2))       # not distinguishable from shape
plt.imshow(data)
plt.xlabel('Time')
plt.ylabel('Channel')
plt.xlim([0, 200])
plt.title('Time-domain signal')
plt.show()
plt.stem(hz, np.mean(2*abs(dataX1[:, :len(hz)]), axis=0), 'k')
plt.xlabel('Frequency (??)')
plt.ylabel('Amplitude')
plt.xlim([0, 200])
plt.title('FFT over channels')
plt.show()
plt.stem(hz, np.mean(2*abs(dataX2[:, :len(hz)]), axis=0), 'k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 200])
plt.title('FFT over time')
plt.show()
=======
# How the FFT works, speed tests
pnts = 1000
signal = np.random.randn(pnts)
# start the timer for "slow" Fourier transform
tic = timeit.default_timer()
fourTime = np.arange(range(pnts))/pnts
fcoefs = np.zeros(len(signal), dtype=complex)
for fi in range(pnts):
	csw = np.exp(-1j*2*np.pi*fi*fourTime)
	fcoefs[fi] = sum(signal, csw)/pnts
toc = timeit.default_timer()
t1 = toc - tic
tic = timeit.default_timer()
fcoefs = scipy.fftpack
>>>>>>> Stashed changes
