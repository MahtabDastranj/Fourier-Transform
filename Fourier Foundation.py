import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import time as ttime
from mpl_toolkits.mplot3d import Axes3D
z = complex(4, 3)
plt.plot(np.real(z), np.imag(z), 'ro')
plt.axis('square')
plt.axis([-5, 5, -5, 5])
plt.grid(True)
plt.xlabel('Real axis'), plt.ylabel('Imaginary axis')
plt.show()
mag = np.abs(z)     #mag = np.sqrt(np.real(z)**2 + np.imag(z)**2)
print(mag)
phs = np.angle(z)   #phs = math.atan(np.imag(Z) / np.real(z))
print(phs)
x = np.linspace(-3, 3, num=50)
plt.plot(x, np.exp(x))
plt.axis([min(x), max(x), 0, np.exp(x[-1])])
plt.grid(True)
plt.xlabel('x')
plt.show()
k = 2/np.pi     #define k (any real number)
Euler = np.exp(1j*k)
plt.plot(np.cos(k), np.sin(k), 'ro')        #plot dot   #r is for red & o is to draw a marker as a circle
x = np.linspace(-np.pi, np.pi, num=200)
plt.plot(np.cos(x), np.sin(x), 'ro')        #deaw unit circle for refrance
#plot_syntaxs :[
#	'r--' = 'red dashes';
#	'bs' = 'blue squares';
#	'g^' = 'green triangles';
#]    o is to macke a circle
plt.axis('square')
plt.grid(True)
plt.xlabel('Real axis'), plt.ylabel('Imaginary axis')
plt.show()
m = 4       #magnitute
k = np.pi/3     #phase
compnum = m*np.exp(1j*k)        #Euler's formula with arbitrary vector magnitude'
mag = np.abs(compnum)
phs = np.angle(compnum)
plt.polar([phs, phs], [0, mag])     #Extract magnitude and angle
plt.show()
srate = 500     #sampling rate in Hz
time = np.arange(0., 2., 1./srate)      #times in second
freq = 3
ampl = 2
phas = np.pi/3
sinewave = ampl * np.sin(2* np.pi * freq * time + phas)
coswave = ampl * np.cos(2* np.pi * freq * time + phas)
plt.plot(time, sinewave, 'k', time, coswave, 'r')
plt.xlabel('Time (sec.)'), plt.ylabel('Amplitude (a.u.)')
plt.title('A sine and cosine with the same parameters.')
plt.show()
freq = 5
csw = ampl * np.exp(1j* (2* np.pi * freq * time + phas))
plt.plot(time, np.real(csw), 'g', label='real')
plt.plot(time, np.imag(csw), 'r', label='imag')
plt.xlabel('Time (sec.)'), plt.ylabel('Amplitude (a.u.)')
plt.title('Complex sinewave projections')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(time, np.real(csw), np.imag(csw), 'r')
ax.set_title('Complex sine wave in all its 3D glory')
ax.set_xlabel('Time (s)'), ax.set_ylabel('Real part'), ax.set_zlabel('Imag part')
plt.show()
v1 = [1, 2, 3]
v2 = [3, 2, 1]
dp = sum(np.multiply(v1, v2))       #compute the dot pot
print('The dot product is ', dp)
#here if we change freq1 the dot product would be super close to zero (try 86 & even 4.5 which is super close to 5)
#However it's is only true for frequancies in step of 1 or 0.5.This is because of the sampling rate
#if we try 4.35 dp is gonna be larger.
ampl1 = ampl2 = 2
phas1 = phas2 = np.pi/2
##The phase shift definitely affects the dot product.
#Yet there is one feature of the dot product that the phase shift does not affect, which is the distance to the origin.
#That distance remains constant while the phase shifts.
# Because the similarity between the complex sine wave and the signal stays the same.
# The phase shift is accounted for by the complex sine wave having a real part and an imaginary part.
# Distance to the origin is a function of both of those parts.
#Sine waves at different frequencies are orthogonal, meaning their dot product is zero.
#That means the Fourier transform has an orthogonal basis set, which is one of the great features of the FT.
sinewave1 = ampl1 * np.sin(2 * np.pi * freq * time + phas1)
sinewave2 = ampl2 * np.sin(2 * np.pi * freq * time + phas2)
dp = np.dot(sinewave1, sinewave2)
print('dot = ', dp)
theta = 0 * np.pi/4      #phase of signal
srate = 1000
time = np.arange(-1., 1., 1./srate)
sinew = np.sin(2*np.pi * 5 * time + theta)
gauss = np.exp(-time**2 / .1)
signal = np.multiply(sinew, gauss)
sinefrex = np.arange(2., 10., .5)
plt.plot(time, signal)
plt.xlabel('Time (sec.)'), plt.ylabel('Amplitude (a.u.)')
plt.title('Signal')
plt.show()
#plotting the dot point between the signal and  and the sinew for each frequancy
dps = list(range(len(sinefrex)))        #dps = np.zeros(len(sinefrex))
for fi in range(len(dps)):
	sinew = np.sin(2* np.pi* sinefrex[fi]* time)
	dps[fi] = np.dot(signal, sinew) / len(time)
	#imagine you have a signal that is 1 second long,& imagine that the dot product between that signal and a sine wave
	#is 100. Now imagine you repeat the signal so you get a new signal that is 2 seconds long
	# but really it's the original signal just repeated. Now the dot product with the sine wave will be 200.
	# But is this longer signal any more like the sine wave than the first signal?
	# No, they are equally similar to the sine wave, because they are exactly the same signal.
	#That's why you would divide by the total length.
	# With this length-normalization, both signals would have a dot product of 100.
plt.stem(sinefrex, dps)
plt.xlabel('Sine wave frequency (Hz)'), plt.ylabel('Dot product')
plt.title('Dot products with sine waves')
plt.show()
dps = np.zeros(len(sinefrex), dtype=complex)        #returning a lit of 0+0j
for fi in range(len(dps)):
	sinew = np.exp(1j* 2* np.pi* sinefrex[fi]* time)
	dps[fi] = np.abs(np.vdot(sinew, signal) / len(time))
plt.stem(sinefrex, dps)
plt.xlabel('Sine wave frequency (Hz)'), plt.ylabel('Dot product')
plt.title('Dot products with sine waves')
plt.show()
sinew = np.sin(2*np.pi*5*time + theta)
gauss = np.exp( (-time**2) / .1)
signal = np.multiply(sinew,gauss)
sinew = np.sin(2* np.pi* 5* time )
cosnw = np.cos(2* np.pi* 5* time)
dps = np.dot( sinew, signal ) / len(time)
dpc = np.dot( cosnw, signal) / len(time)
dp_complex = complex(dpc, dps)
mag = np.abs(dp_complex)
phs = np.angle(dp_complex)
plt.plot(dpc, dps, 'ro')
plt.xlabel('Cosine axis')
plt.ylabel('Sine axis')
plt.axis('square')
plt.grid(True)
plt.axis([-.2, .2, -.2, .2])
plt.show()
plt.polar([phs, phs], [0, mag])
plt.show()
#ilustration of the effect of phase offsets on dot product
csw = np.exp(1j* 2* np.pi* 5* time)
rsw = np.sin(2* np.pi* 5* time)
phase = np.linspace(0, 7* np.pi/2, num=100)
for phi in range(0,len(phase)):
	sinew = np.sin(2* np.pi* 5* phase[phi] + phase)
	signal = np.multiply(sinew, gauss)
cdp = np.sum(np.multiply(signal, csw))/ len(time)
rdp = sum(np.multiply(signal, rsw))/ len(time)
#plot signal and real part of sine wave
#np.sum(np.multiply()) is just another form of np.dot
pl.cla()        #wipe the figure
plt.subplot2grid((2, 2), (0, 0), colspan=2)
plt.plot(time, signal)
plt.plot(time, rsw)
plt.title('Signal and Sine wave over time')
#plot complex dot product
plt.subplot2grid((2, 2), (1, 0))
plt.plot(np.real(cdp), np.imag(cdp))
plt.xlabel('Real'), plt.ylabel('Imaginary')
plt.axis('square')
plt.grid(True)
plt.axis([-.2, .2, -.2, 2])
plt.plot([0, np.real(cdp)], [0, np.imag(cdp)], 'r')
#plot normal dot product
plt.subplot2grid((2, 2), (1, 1))
plt.plot(rdp, 0, 'ro')
plt.xlabel('Real')
plt.axis('square')
plt.grid(True)
plt.axis([-.2, .2, -.2, 2])
display.clear_output(wait=True)
display.display(pl.gcf())       #gcf is to get the current figure
ttime.sleep(0.01)       #creating time delay
