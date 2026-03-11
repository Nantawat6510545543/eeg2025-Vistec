import numpy as np
from scipy.signal import welch, butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Filters EEG into specific rhythms (Theta, Alpha, Beta, Gamma)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def extract_hjorth_parameters(signal):
    """Calculates Activity, Mobility, and Complexity (Features 4, 5, 6)"""
    # Activity: Variance of the signal
    activity = np.var(signal)
    
    # Mobility: Standard deviation of slope / standard deviation of amplitude
    diff_1 = np.diff(signal)
    mobility = np.sqrt(np.var(diff_1) / activity)
    
    # Complexity: Mobility of first derivative / Mobility of signal
    diff_2 = np.diff(diff_1)
    mobility_diff_1 = np.sqrt(np.var(diff_2) / np.var(diff_1))
    complexity = mobility_diff_1 / mobility
    
    return activity, mobility, complexity

def extract_time_freq_features(signal, fs):
    """Extracts the 9 features mentioned in Table 1 of the paper"""
    features = {}
    
    # 1. Peak-Peak Mean
    features['peak_to_peak'] = np.mean(np.ptp(signal))
    
    # 2. Mean Square Value
    features['msv'] = np.mean(signal**2)
    
    # 3. Variance
    features['variance'] = np.var(signal)
    
    # 4-6. Hjorth Parameters
    act, mob, comp = extract_hjorth_parameters(signal)
    features['hjorth_activity'] = act
    features['hjorth_mobility'] = mob
    features['hjorth_complexity'] = comp
    
    # 7-9. Frequency domain features (using PSD)
    freqs, psd = welch(signal, fs, nperseg=len(signal))
    features['max_psd_freq'] = freqs[np.argmax(psd)]
    features['max_psd_value'] = np.max(psd)
    features['power_sum'] = np.sum(psd)
    
    return features

import numpy as np
import antropy as ent

def extract_nonlinear_features(signal):
    """Extracts non-linear dynamical features (Features 10-18)"""
    nonlinear_feats = {}
    
    # 10. Approximate Entropy
    nonlinear_feats['approx_entropy'] = ent.app_entropy(signal)
    
    # 12. Correlation Dimension (Sample Entropy is a common proxy)
    nonlinear_feats['sample_entropy'] = ent.sample_entropy(signal)
    
    # 15. Permutation Entropy
    nonlinear_feats['perm_entropy'] = ent.perm_entropy(signal, normalize=True)
    
    # 16. Singular Entropy (SVD Entropy)
    nonlinear_feats['svd_entropy'] = ent.svd_entropy(signal, normalize=True)
    
    # 17. Shannon Entropy
    # First, convert to a probability distribution (histogram)
    hist, _ = np.histogram(signal, bins=10, density=True)
    hist = hist[hist > 0] # Remove zeros for log calculation
    nonlinear_feats['shannon_entropy'] = -np.sum(hist * np.log2(hist))
    
    # 18. Spectral Entropy
    nonlinear_feats['spectral_entropy'] = ent.spectral_entropy(signal, sf=128, method='welch', normalize=True)
    
    return nonlinear_feats