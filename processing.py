import numpy as np
from obspy.core import UTCDateTime
from scipy import signal

def detrend(data, type='linear'):
    """
    function to detrend data
    data: numpy array of data
    type: type of detrending, now only linear and simple
    returns detrended data
    """
    if type == 'linear':
        data = signal.detrend(data)
    
    if type == 'simple':
        for i, tr in enumerate(data):
            ndat = len(tr)
            x1, x2 = tr[0], tr[-1]
            if x1 == 0:
                if x2 == 0:
                    break
            tr -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
            data[i] = tr.copy()
    return data

def taper(data, percentage):
    """
    funtion to taper data
    data: numpy array of data
    percentage: fraction of data to taper, between 0 and 1
    returns tapered data
    """
    width=int(len(data)*percentage)
    
    if len(data.shape) == 2:
        nt=data.shape[1]
        for i in range(width): 
            data[:,i]=(float(i+1)/float(width+1))*data[:,i]
            data[:,nt-i-1]=(float(i+1)/float(width+1))*data[:,nt-i-1]
            
    elif len(data.shape) == 1:
        nt=data.shape[0]
        for i in range(width): 
            data[i]=(float(i+1)/float(width+1))*data[i]
            data[nt-i-1]=(float(i+1)/float(width+1))*data[nt-i-1]
    
    return data

def trim(data, start, end, t0, fs=200):
    """
    function to trim data
    data: numpy array of data
    start: UTCDateTime object of desired start time
    end: UTCDateTime object of desired end time
    meta: metadata file
    returns trimmed data
    """
    
    trim0 = int((start - t0) * fs)
    trim1 = int((end - t0) * fs)
    
    return data[:,trim0:trim1]


def zerophase_chebychev_lowpass_downsamp(data, fs, factor):
    """
    Custom Chebychev type two zerophase lowpass filter useful for
    decimation filtering.

    This filter is stable up to a reduction in frequency with a factor of
    10. If more reduction is desired, simply decimate in steps.

    Partly based on a filter in ObsPy.

    :param trace: The trace to be filtered.
    :param freqmax: The desired lowpass frequency.
    """
    freqmax = fs/factor
    #print("Downsampling {0}Hz to {1}Hz".format(fs,freqmax))
    
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freqmax / (fs * 0.5)  # stop band frequency
    wp = ws  # pass band frequency

    while True:
        if order <= 12:
            break
        wp *= 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

    b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

    data2 = np.zeros([  np.int(np.ceil(np.shape(data)[0]/factor)), np.shape(data)[1]])
    for i in range(np.shape(data)[1]):
        y = filtfilt(b,a,data[:,i])
        sl = [slice(None)]*y.ndim
        sl[-1]=slice(None,None,factor)
        data2[:,i] = y[tuple(sl)]
    #print("   Downsampling completed.")
    return data2
    
def block_bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter. Taken directly from OBSPY
    

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    #print("Filtering {0}Hz to {1}Hz".format(freqmin,freqmax))
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = signal.iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data, axis=0 )
        return signal.sosfilt(sos, firstpass[::-1], axis=0)[::-1]
    else:
        return signal.sosfilt(sos, data, axis=0)
    
    