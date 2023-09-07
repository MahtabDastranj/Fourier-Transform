# 1.fourtime
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import scipy.fftpack
import time
# The DTFT in loop form
# create the signal
srate = 1000    # hz
time = np.arange(0., 2., 1/srate)
pnts = len(time)        # number of time points
signal = 2.5 * np.sin(2*np.pi*4*time) + 1.5 * np.sin(2*np.pi*6.5*time)
fourTime = np.array(range(pnts)) / pnts
fCoefs = np.zeros(len(signal), dtype=complex)       # fourier coefficient is the complex dot product
for fi in range(pnts):
	# create complex sine wave
	csw = np.exp(-1j * 2 * np.pi * fi * fourTime)
	#  It's related to the relationship between the forward and inverse Fourier transform.
	#  A real-valued signal needs to be real-valued after applying the inverse FT
	#  so the imaginary components need to cancel each other.
	#  Having them positive in one direction and negative in the other direction is the mechanism of cancellation.
	#  You could use +i in the forward FT and then -i in the inverse FT.
	#  The standard way to do it (- forward, + inverse).
	# compute dot product between complex sine wave and signal which is called Fourier coefficient
	fCoefs[fi] = np.vdot(signal, csw) / pnts
	
ampl = 2 * np.abs(fCoefs)       # showing the distance from the origin
hz = np.linspace(0, srate/2, int(pnts/2.) + 1)
# srate/2 is referring to the nyquist
# So the N/2+1 comes from N/2 positive frequencies starting here, plus one extra for the DC
# stem plot should always be used for showing the result of fourier transform
plt.stem(hz, ampl[range(len(hz))])
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
plt.xlim(0, 10)
plt.show()
coefs2plot = [0, 0]
coefs2plot[0] = np.argmin(np.abs(hz - 4))
coefs2plot[1] = np.argmin(np.abs(hz - 4.5))
#  identifies the index in the hz vector with values closest to 4 and 4.5
mag = np.abs(fCoefs[coefs2plot])
phs = np.angle(fCoefs[coefs2plot])
plt.plot(np.real(fCoefs[coefs2plot]), np.imag(fCoefs[coefs2plot]), 'o', linewidth=2, markersize=10, markerfacecolor='r')
# The more the distance from origin is in the plot the more similar they are
plt.plot([-2, 2], [0, 0], 'k', linewidth=2)     # vertical axes :black
plt.plot([0, 0], [-2, 2], 'k', linewidth=2)     # horizontal axes :black
axislims = np.max(mag)*1.1
plt.grid()
plt.axis('square')
plt.xlim([-axislims, axislims])
plt.ylim([-axislims, axislims])
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.title('Complex plane')
plt.show()
pnts = 16
fourTime = np.array(range(pnts)) / pnts
# [0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375]
for fi in range(pnts):
	# create complex sine wave
	csw = np.exp(-1j * 2 * np.pi * fi * fourTime)
	# and plot it
	loc = np.unravel_index(fi, [4, 4], 'F')
	plt.subplot2grid((4, 4), (loc[1], loc[0]))
	plt.plot(fourTime, np.real(csw))
	plt.plot(fourTime, np.imag(csw))
plt.show()
srate = 1000
npnts = 100001
if npnts % 2 == 0:
	topfreq = srate / 2
else:
	topfreq = srate/2 * (npnts-1)/npnts
hz1 = np.linspace(0, srate/2, np.floor(npnts/2+1).astype(int))
hz2 = np.linspace(0, topfreq, np.floor(npnts/2+1).astype(int))
n = 16
print('%.9f\n%.9f' % (hz1[n], hz2[n]))
# no difference if npnts is even, otherwise there is a tiny difference which minimize as npnts grows
# Case1 ,Odd number of data points, N is correct
srate = 1000
time = np.arange(0, srate + 1)/ srate               # dividing by sampling rate converts it to seconds
npnts = len(time)
signal = np.sin(15 * 2 * np.pi * time)
# scipy.fftpack.fft(x, n=None, axis=-1, overwrite_x=False)[source]
# Return discrete Fourier transform of real or complex sequence
# signal's amplitude spectrum
signalX = 2 * np.abs(scipy.fftpack.fft(signal)) / len(signal)
# frequencies vectors
hz1 = np.linspace(0, srate, npnts + 1)
hz2 = np.linspace(0, srate, npnts)
fig = plt.subplots(1, figsize=(10, 15))
plt.plot(hz1[:npnts], signalX, 'bo', label='N+1')
plt.plot(hz2, signalX, 'rs', label='N')
plt.legend()
plt.xlim(14.9, 15.1)
plt.ylim(.99, 1.01)
plt.xlabel('Frequency (Hz)')
plt.title(str(len(time)) + ' points long')
plt.ylabel('Amplitude')
plt.show()
# Case 2: Even number of data points, N+1 is correct
srate = 1000
time = np.arange(0, srate)/ srate
npnts = len(time)
signal = np.sin(15 * 2 * np.pi * time)
signalX = 2 * np.abs(scipy.fftpack.fft(signal)) / len(signal)
hz1 = np.linspace(0, srate, npnts + 1)
hz2 = np.linspace(0, srate, npnts)
fig = plt.subplots(1, figsize=(10, 15))
plt.plot(hz1[:npnts], signalX, 'bo', label='N+1')
plt.plot(hz2, signalX, 'rs', label='N')
plt.legend()
plt.xlim(14.9, 15.1)
plt.ylim(.99, 1.01)
plt.xlabel('Frequency (Hz)')
plt.title(str(len(time)) + ' points long')
plt.ylabel('Amplitude')
plt.show()
# Case 3: longer signal
srate = 1000
time = np.arange(0, 5*srate)/srate
npnts = len(time)
signal = np.sin(15 * 2 * np.pi * time)
signalX = 2*np.abs(scipy.fftpack.fft(signal)) / len(signal)
hz1 = np.linspace(0, srate, npnts+1)
hz2 = np.linspace(0, srate, npnts)
fig = plt.subplots(1, figsize=(10, 5))
plt.plot(hz1[:npnts], signalX, 'bo', label='N+1')
plt.plot(hz2, signalX, 'rs', label='N')
plt.xlim([14.99, 15.01])         # If you zoom out(0, 10) you'd assume there's no difference between 2 signals
plt.ylim([.99, 1.01])
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.title(str(len(time)) + ' points long')
plt.ylabel('Amplitude')
plt.show()
# Section : Normalized time vector
srate = 1000
time = np.arange(0, 2 * srate)/srate
# or time = np.arange(0, 2*srate, 1/srate)
# Both ways are to convert time vectors into seconds
pnts = len(time)
signal = 2.5 * np.sin(2*np.pi*4*time) + 1.5 * np.sin(2*np.pi*6.5*time)
plt.plot(time, signal, 'k')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.show()
# Prepare the fourier transform
fourTime = np.arange(0, pnts)/pnts
fCoefs = np.zeros(len(signal), dtype=complex)       # len(signal) = 2000
for fi in range(pnts):
	csw = np.exp(-1j * 2 * np.pi * fi * fourTime)
	fCoefs[fi] = sum(signal * csw)/pnts         # sum(np.multiply(signal, csw))/pnts
	# We /pnts to normalize fcoefs otherwise the ampl shown would be much higher than what it should actually be
	# Plus the amplitude would rise as the signal gets longer by time. Called Normalizing
ampls = 2 * abs(fCoefs)
# It's multiplied by 2 because fcoef represents only the positive frequencies
# since the amplitude of negative frequencies is the mirror of positive ones we use *2
# In this case we haven't considered the frequency = 0 because it holds zero amplitude(energy)
hz = np.linspace(0, srate/2, int(pnts/2+1))
plt.stem(hz, ampls[:len(hz)], 'ks-')
plt.xlim([0, 10])
plt.ylim([-.01, 3])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Amplitude spectrum')
plt.show()
plt.stem(hz, np.angle(fCoefs[:len(hz)]), 'ks-')
plt.xlim([0, 10])
plt.ylim([-np.pi, np.pi])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad.)')
plt.title('Phase spectrum')
plt.show()
# Reconstructing time signal
reconTS = np.real(scipy.fftpack.ifft(fCoefs)) * pnts
plt.plot(time, signal, 'k', label='Original')
plt.plot(time[::3], reconTS[::3], 'r.', markersize=1, label='Reconstructed')
# if fourtime is (n, pnts+n) there'd be a time shift making the amplitude look exactly the same
# And the phase spectrum with a slight change
plt.legend()
plt.show()
# averaging fourier coefficient
ntrials = 100
srate = 200
time = np.arange(0, 1, 1/srate)
pnts = len(time)
data_set = np.zeros((ntrials, pnts))
for triali in range(ntrials):
	data_set[triali, :] = np.sin(2 * np.pi * 20 * time + 2 * np.pi * np.random.rand())
	#  change the 2 in phase shift, you'd recognize they become more alike as 2 lowers
	# rand() picks either 0 or 1
fig, axs = plt.subplots(2)
for i in range(ntrials):
	axs[0].plot(time, data_set[i, :])
axs[0].plot(time, np.mean(data_set, axis=0), 'k', linewidth=3)
# The average is lower than each because of the phase difference
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.xlim([0, .1])
# Get the fourier coefficient
data_setX = scipy.fftpack.fft(data_set, axis=1)
hz = np.linspace(0, srate/2, int(pnts/2)+1)
# averaging option 1: complex Fourier coefficients, then magnitude
ave1 = 2 * np.abs(np.mean(data_setX, axis=0))
# averaging option 2: magnitude, then complex Fourier coefficients
ave2 = np.mean(2 * np.abs(data_setX), axis=0)
axs[1] = plt.stem(hz, ave1[0:len(hz)], 'ks-', label='Average coefficients', use_line_collection=True)
axs[1] = plt.stem(hz+.2, ave2[0:len(hz)], 'ro-', label='Average amplitude', use_line_collection=True)
# We understand that as the phase shift decreases the Average coefficients gets closer to the average amplitude
# using frequency shift to make the plot identifiable
plt.xlim([10, 30])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency domain')
plt.legend()
plt.tight_layout()
plt.show()
# DC reconstruction without careful normalization
srate = 1000
time = np.arange(0., 2., 1/srate)
pnts = len(time)
signal = 1.5 + 2.5*np.sin(2*np.pi*4*time)
# Preparing the fourier transform
fourTime = np.array(range(pnts))/pnts
fCoefs = np.zeros(len(signal), dtype=complex)
for fi in range(pnts):
	csw = np.exp(-1j*2*np.pi*fi*fourTime)
	fCoefs[fi] = sum(np.multiply(signal, csw))/pnts
hz = np.linspace(0, srate/2, int(np.floor(pnts/2)+1))
ampls = np.zeros(len(signal), dtype=int)
ampls = abs(fCoefs)
ampls[1:] = 2 * abs(fCoefs[1:])
# Indicates first frequency index(after 0)
plt.stem(hz, ampls[0: len(hz)])
plt.xlim(-.1, 10)
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
plt.show()
# Amplitude spectrum vs. power spectrum
srate = 1000
time = np.arange(0, .85, 1/srate)
pnts = len(time)
signal = 2.5*np.sin(2*np.pi*10*time)
signalX = scipy.fftpack.fft(signal)/pnts
hz = np.linspace(0, srate/2, int(pnts/2)+1)
signalAmp = abs(signalX)
signalAmp[1:] = 2*abs(signalX[1:])
signalPow = signalAmp**2
# There's conservation of energy in both time and frequency domain
plt.figure(figsize=(12, 3))
plt.subplot2grid((1, 3), (0, 0))
plt.plot(time, signal)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.subplot2grid((1, 3), (0, 1))
plt.plot(hz, signalAmp[0:len(hz)], 'ks-', label='Amplitude')
plt.plot(hz, signalPow[0:len(hz)], 'rs-', label='Power')
plt.xlim([0, 20])
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude or power')
plt.title('Frequency domain')
plt.subplot2grid((1, 3), (0, 2))
plt.plot(hz, 10*np.log10(signalPow[0: len(hz)]), 'ks-')
plt.xlim([0, 20])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decibel power')
plt.tight_layout()
plt.show()
