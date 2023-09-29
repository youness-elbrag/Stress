
# signal processing
from scipy import signal
import numpy as np 
from scipy.ndimage import label
from scipy.stats import zscore
from scipy.interpolate import interp1d, splrep, splev
from scipy.integrate import trapz
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt 

def Template_Matching(ecg_signal, threshold=0.3, qrs_filter=None):
    '''
    Detects ECG R-peaks using cross-correlation and thresholding.
    
    Parameters:
    - ecg_signal (pd.Series): The ECG signal data as a pandas Series.
    - threshold (float): The threshold value for peak detection.
    - qrs_filter (np.array): The QRS filter for cross-correlation.
    
    Returns:
    - tuple: A tuple containing two arrays. The first array holds the indices of detected peaks.
             The second array contains the similarity values from cross-correlation.
    '''
    if qrs_filter is None:
        # Create a default QRS filter, which is a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    
    # Normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # Calculate cross-correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # Return peak indices and similarity values using threshold
    peak_indices = np.where(similarity > threshold)[0]
    return peak_indices, similarity
    
def get_plot_ranges(start=10, end=20, n=5):
    '''
    Make an iterator that divides into n or n+1 ranges. 
    - if end-start is divisible by steps, return n ranges
    - if end-start is not divisible by steps, return n+1 ranges, where the last range is smaller and ends at n
    
    # Example:
    >> list(get_plot_ranges())
    >> [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]

    '''
    distance = end - start
    for i in np.arange(start, end, np.floor(distance/n)):
        yield (int(i), int(np.minimum(end, np.floor(distance/n) + i)))


# Interpolate and compute HR
def interp_cubic_spline(rri, sf_up=4):
    """
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.
    
    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.
    
    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
    rri_time = np.cumsum(rri) / 1000.0
    time_rri = rri_time - rri_time[0]
    time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
    tck = splrep(time_rri, rri, s=0)
    rri_interp = splev(time_rri_interp, tck, der=0)
    return rri_interp


def group_peaks(p, threshold):
    '''
    The peak detection algorithm finds multiple peaks for each QRS complex. 
    Here we group collections of peaks that are very near (within a threshold) and we take the median index 
    '''
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output

def timedomain(rr):
    hr = 60000/rr
    
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    mean_hr_kubios = 60000/mean_rr
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    min_hr = np.min(hr)
    max_hr = np.max(hr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    nnxx = np.sum(np.abs(np.diff(rr)) > 50) * 1
    pnnxx = 100 * nnxx / len(rr)

    return np.array([mean_rr, std_rr, mean_hr_kubios, mean_hr, std_hr, min_hr, max_hr, rmssd, nnxx, pnnxx])

def frequency_domain(rri, fs=4):
    fxx, pxx = signal.welch(x=rri, fs=fs)

    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = trapz(pxx[cond_lf], fxx[cond_lf])
    hf = trapz(pxx[cond_hf], fxx[cond_hf])
    total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)

    return np.array([vlf, lf, hf, total_power, lf_hf_ratio, peak_vlf, peak_lf, peak_hf, lf_nu, hf_nu])


def RR_interpolate_HR(signal_type, data, threshold=0.7):
    """
    Compute heart rate (HR) from RR intervals or HR values.

    Parameters:
        signal_type (str): Either 'HR' for heart rate signal or 'RR' for RR interval signal.
        data (array-like): Array containing either HR values or RR intervals.
        threshold (float): Similarity threshold for peak detection in template matching.

    Returns:
        np.array: Array containing mean HR and label (0 or 1) based on certain conditions.
    """

    sf_up = 4
    hr = None

    if signal_type == "HR":
        peaks, similarity = template_matching(data, threshold)
        grouped_peaks = group_peaks(peaks)
        rri = np.diff(grouped_peaks)
        rri_interp = interp_cubic_spline(rri, sf_up)
        hr = 1000 * (60 / rri_interp)
    elif signal_type == "RR":
        sf = 1  # Sample frequency (replace with actual value)
        rr = (data / sf) * 1000
        rri = np.diff(rr)
        rri_interp = interp_cubic_spline(rri, sf_up)
        hr = 1000 * (60 / rri_interp)

    if hr is not None:
        mean_hr = np.mean(hr)
        label = 0 if mean_hr < 60 or mean_hr > 100 else 1
        return np.array([mean_hr, label])

    return "None of the Signals are Computed HR Bpm"

def generate_and_save_spectrogram(rr_intervals: np.ndarray, output_file: str):
    """Generate and save a spectrogram from RR interval data."""
    # Calculate RR intervals in seconds
    rr_seconds = rr_intervals / 1000.0
    
    # Compute the spectrogram using scipy.signal.spectrogram
    plt.specgram(x=rr, Fs=1, NFFT=30, noverlap=28, cmap="twilight")

    #f, t, Sxx = signal.spectrogram(rr_seconds, fs=1.0, nperseg=256)
    #plt.imshow(10 * np.log10(Sxx), cmap='twilight', origin='lower', aspect='auto')
    
    plt.axis('off')  # Turn off axis
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
    
    plt.close()  # Close the figure to free up resources
    
    print(f"Spectrogram saved as {output_file}")