import math
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from scipy.misc import derivative
from scipy.fft import fft

def filterJointSignal(joint2dCoords, filterOrd=3, f_c=2.5, f_s=30):
    #extract positions from tuple
    posX, posY = zip(*joint2dCoords)
    #filter signals
    b, a = signal.butter(filterOrd, f_c, fs=f_s)
    dfX = signal.filtfilt(b, a, posX)
    dfY = signal.filtfilt(b, a, posY)
    return dfX, dfY

def deriveSignal(signal_in, dvRange=10):
    dSig=np.zeros(len(signal_in))
    def vfunSig(x):
        return signal_in[int(x)]
    for i in range(len(signal_in)-dvRange):
        dSig[i]=derivative(vfunSig, i, dx=dvRange)
    return dSig

def calcJointSignals(joint2dCoords, filterOrd=3, f_c=2.5, f_s=30, dvRange=10):
    #extracts signals and apply filter
    dfX, dfY = filterJointSignal(joint2dCoords, filterOrd,f_c,f_s)
    #virtual functions for derivatives
    def vfunX(x):
        return dfX[int(x)]
    def vfunY(x):
        return dfY[int(x)]
    #derivatives 
    dX=np.zeros(len(dfX))
    for i in range(len(dfX)-dvRange):
        dX[i]=derivative(vfunX, i, dx=dvRange)
    dY=np.zeros(len(dfY))
    for i in range(len(dfY)-dvRange):
        dY[i]=derivative(vfunY, i, dx=dvRange)
    dMod=[math.sqrt(x**2+y**2) for x,y  in zip(dX,dY)]
    dArg=[ 0 if (x==0) else math.atan(y/x) for x,y  in zip(dX,dY)]
    return dfX, dfY, dX, dY, dMod, dArg

def calcfftPower(signal_in, N):
    out_fft=fft(signal_in)
    pow_fft=2.0/N * np.abs(out_fft[0:N//2])
    return pow_fft

def getMovFftPeaks(FFT_Signal ,step_freq=1, max_peaks=10, perc_sens=0.09):
    peaksArr=np.full((max_peaks), np.nan)
    #sensibility as percentage of original signal 0.09 detects peaks of 9% alt.
    peaks, _ = find_peaks(FFT_Signal, height=(perc_sens*max(FFT_Signal)))
    if len(peaks>max_peaks):#max number of peaks
        peaks=peaks[0:max_peaks]
    peaksArr[0:len(peaks)]=peaks
    return np.dot(peaksArr, step_freq) #frequency per sample