# source: https://stackoverflow.com/questions/19975030/amplitude-of-numpys-fft-results-is-to-be-multiplied-by-sampling-period

import numpy as np
import matplotlib.pyplot as plt

# calculate amplitude of fft
# parameters:
#  N: length of signal
#  t: vector with time stamps
#  f: signal in time domain
# returns:
#  freq: frequencies of FFT
#  fft_a: real values of FFT, scaled to represent the amplitude of the signal in the time domain
def fft_amplitude(N,t,f):
    # time difference between two samples
    dt=t[1]-t[0]
    freq=np.fft.rfftfreq(N, dt)
    # calculate fft
    ft=np.fft.rfft(f)
    magnitude=np.abs(ft)
    fft_a=magnitude/N*2.0
    fft_a[0]=fft_a[0]/2.0 # frequency 0 and nyquist frequency must be divided by 2
    fft_a[-1]=fft_a[-1]/2.0
    return freq,fft_a

# plot signal in time and frequency domain
# parameters
#  t: vector with time stamps
#  f: signal in time domain
#  freq: frequencies of FFT
#  fft_a: real values of FFT, scaled to represent the amplitude of the signal in the time domain
def plot(t,f,freq,fft_a,title="")->None:
    plt.subplot(2,1,1)
    plt.plot(t, f)
    plt.xlabel('time/s')
    plt.ylabel('amplitude/(m/s^2)')
    plt.subplot(2,1,2)
    plt.plot(freq, fft_a,'x-')
    plt.xlabel('f/Hz')
    plt.ylabel('amplitude/(m/s^2)')
    #plt.xlim([0, 1.4])
    plt.title(title)
    plt.show()

# 1) signal length 4096, time from -50 s till +50 s
#    sinusoidal signal with amplitude A=5.0, frequency f=1 Hz and offset=0 m/s^2 (Gleichanteil, mean value)
N=4096
T=100.0
A=5.0
t=np.linspace(-T/2,T/2,N)
f=A*np.sin(2*np.pi*t)
freq,fft_a=fft_amplitude(N,t,f)
plot(t,f,freq,fft_a,'A=5.0, f=1 Hz, offset=0')

# 2) signal length 512, time from 0 s till 512/sample rate=0,07683 s
#    sinusoidal signal with amplitude A=5.0, frequency f=666.4 Hz and offset=10 m/s^2 (Gleichanteil, mean value)
N=512
sample_rate=6664.0 # Hz
A=5.0
m=10.0
t=np.arange(0,N)/sample_rate
sf=666.4 # signal frequency
f=A*np.sin(2*np.pi*sf*t)+m
freq,fft_a=fft_amplitude(N,t,f)
plot(t,f,freq,fft_a,'N=512, A=5.0, f=666.4 Hz, sample rate=6664 Hz, offset=10')

# 3) signal length 512, time from 0 s till 512/sample rate=0,07683 s
#    - 1st sinusoidal signal with amplitude A=5.0, frequency f=666.4 Hz
#    - 2nd sinusoidal signal with amplitude A=10.0, frequency f=50 Hz
#    - offset=15 m/s^2 (Gleichanteil, mean value)
N=512
sample_rate=6664.0 # Hz
m=15.0
t=np.arange(0,N)/sample_rate
A1=5.0
sf1=666.4 # signal frequency
f1=A1*np.sin(2*np.pi*sf1*t)
A2=10.0
sf2=50.0 # signal frequency
f2=A2*np.sin(2*np.pi*sf2*t)
f=f1+f2
f=f+m
freq,fft_a=fft_amplitude(N,t,f)
plot(t,f,freq,fft_a,'N=512, A1=5.0, f1=666.4 Hz, A2=10.0, f2=50 Hz, sample rate=6664 Hz, offset=15')

# 4) signal length 512, time from 0 s till 512/sample rate=0,07683 s
#    - 1st sinusoidal signal with amplitude A=5.0, frequency f=559.671875 Hz
#    - 2nd sinusoidal signal with amplitude A=10.0, frequency f=52.0625 Hz
#    - offset=15 m/s^2 (Gleichanteil, mean value)
# Each of the frequencies of both signals matches exactly the frequency of an FFT bin --> no "spectral leakage"
N=512
sample_rate=6664.0 # Hz
m=15.0
t=np.arange(0,N)/sample_rate
A1=5.0
sf1=559.671875 # signal frequency
f1=A1*np.sin(2*np.pi*sf1*t)
A2=10.0
sf2=52.0625 # signal frequency
f2=A2*np.sin(2*np.pi*sf2*t)
f=f1+f2
f=f+m
freq,fft_a=fft_amplitude(N,t,f)
print(freq)
plot(t,f,freq,fft_a,'N=512, A1=5.0, f1=559.67 Hz, A2=10.0, f2=52.0625 Hz, sample rate=6664 Hz, offset=15')

# 5) signal length 512, time from 0 s till 512/sample rate=0,07683 s
#    - sinusoidal signal with amplitude A=5.0, frequency f=3332 Hz (Nyquist frequency)
#    - offset=0 m/s^2 (Gleichanteil, mean value)
# Test for maximal frequency (Nyquist frequency)
N=512
sample_rate=6664.0 # Hz
t=np.arange(0,N)/sample_rate
A1=5.0
sf1=3332.0 # signal frequency, Nyquist frequency
f1=A1*np.sin(2*np.pi*sf1*t+np.pi/2.0) # phase shift by 90°
freq,fft_a=fft_amplitude(N,t,f1)
print(freq)
plot(t,f1,freq,fft_a,'N=512, A1=5.0, f1=3332 Hz, phase shift=90°, Niquist frequency')
