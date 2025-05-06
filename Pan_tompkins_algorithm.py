######################################
# Pan–Tompkins algorithm in python   #
# Author: Pramod kumar Chinthala     #
#                                    #
######################################

import numpy as np
import csv
from scipy.signal import butter, filtfilt, find_peaks

class Pan_tompkins:
    """ Implementationof Pan Tompkins Algorithm.

    Noise cancellation (bandpass filter) -> Derivative step -> Squaring and integration.

    Params:
        data (array) : ECG data
        sampling rate (int)
    returns:
        Integrated signal (array) : This signal can be used to detect peaks


    ----------------------------------------
    HOW TO USE ?
    Eg.

    ECG_data = [4, 7, 80, 78, 9], sampling  =2000
    
    call : 
       signal = Pan_tompkins(ECG_data, sampling).fit()

    ----------------------------------------
    
    """
    def __init__(self, data, sample_rate):

        self.data = data
        self.sample_rate = sample_rate


    def fit(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150, window_size=None):
        ''' Fit the signal according to algorithm and returns integrated signal
        
        '''
        # 1.Noise cancellationusing bandpass filter
        self.filtered_BandPass = self.band_pass_filter(normalized_cut_offs, butter_filter_order, padlen)
        
        # 2.derivate filter to get slpor of the QRS
        self.derviate_pass = self.derivative_filter()

        # 3.Squaring to enhance dominant peaks in QRS
        self.square_pass = self.squaring()

        # 4.To get info about QRS complex
        self.integrated_signal = self.moving_window_integration( window_size)

        # 5.Peak detection
        self.peak_signal = self.peak_detection()

        return self.integrated_signal


    def band_pass_filter(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150):
        ''' Band pass filter for Pan tompkins algorithm
            with a bandpass setting of 5 to 20 Hz

            params:
                normalized_cut_offs (list) : bandpass setting canbe changed here
                bandpass filte rorder (int) : deffault 2
                padlen (int) : padding length for data , default = 150
                        scipy default value = 2 * max(len(a coeff, b coeff))

            return:
                filtered_BandPass (array)
        '''

        # Calculate nyquist sample rate and cutoffs
        nyquist_sample_rate = self.sample_rate / 2

        # calculate cutoffs
        if normalized_cut_offs is None:
            normalized_cut_offs = [5/nyquist_sample_rate, 15/nyquist_sample_rate]
        else:
            assert type(self.sample_rate ) is list, "Cutoffs should be a list with [low, high] values"

        # butter coefficinets 
        b_coeff, a_coeff = butter(butter_filter_order, normalized_cut_offs, btype='bandpass')[:2]

        # apply forward and backward filter
        filtered_BandPass = filtfilt(b_coeff, a_coeff, self.data, padlen=padlen)
        
        return filtered_BandPass


    def derivative_filter(self):
        ''' Derivative filter

        params:
            filtered_BandPass (array) : outputof bandpass filter
        return:
            derivative_pass (array)
        '''

        # apply differentiation
        derviate_pass= np.diff(self.band_pass_filter())

        return derviate_pass


    def squaring(self):
        ''' squaring application on derivate filter output data

        params:

        return:
            square_pass (array)
        '''

        # apply squaring
        square_pass= self.derivative_filter() **2

        return square_pass 


    def moving_window_integration(self, window_size=None):
        ''' Moving avergae filter 

        Params:
            window_size (int) : no. of samples to average, if not provided : 0.08 * sample rate
            sample_rate (int) : should be given if window_size is not given  
        return:
            integrated_signal (array)
        '''

        if window_size is None:
            assert self.sample_rate is not None, "if window size is None, sampling rate should be given"
            window_size = int(0.08 * int(self.sample_rate))  # given in paper 150ms as a window size
        

        # define integrated signal
        integrated_signal = np.zeros_like(self.squaring())

        # cumulative sum of signal
        cumulative_sum = self.squaring().cumsum()

        # estimationof area/ integral below the curve deifnes the data
        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)

        return integrated_signal
    
    def peak_detection(self):
        integrated_signal = self.moving_window_integration()
        peaks, _ = find_peaks(integrated_signal, height=np.mean(integrated_signal), width=0.2, distance=self.sample_rate*0.2)
        return peaks

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import wfdb
    import argparse

    parser = argparse.ArgumentParser("Pan-Tompkins implementation in python")
    parser.add_argument("-i", "--input", required=True, 
                        help="ECG data input, file extension must be excluded")
    args = parser.parse_args()

    ECG_data = []
    timeStamp = []

    data, fields = wfdb.rdsamp(args.input, channels=[0])
    ECG_data = []        
    for d in data: # Get data of channel 0
        ECG_data.append(d[0])

    timeStamp = [i/fields["fs"] for i in range(len(ECG_data))]
    
    dft = np.diff(timeStamp)
    av = np.average(dft)
    print(av)

    algo = Pan_tompkins(ECG_data, 100)
    signal = algo.fit()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    ax1.plot(timeStamp, ECG_data)
    ax1.set_title("Raw ECG")
    ax1.grid(True)

    ax2.plot(timeStamp[:-1], signal)
    
    peak_index = algo.peak_detection()
    peak_timeStamp = [timeStamp[i] for i in peak_index]

    for peakTime in peak_timeStamp:
        plt.axvline(x=peakTime, color='red', linestyle='-', label='Seuil = 6')

    BPM = 60 / np.mean(np.diff(peak_timeStamp))
    print(f"BPM={BPM} ")

    ax2.set_title("Filtered ECG")
    ax2.grid(True)

    plt.show()
