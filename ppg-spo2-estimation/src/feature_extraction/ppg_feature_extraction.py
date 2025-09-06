"""
PPG Feature Extraction and Beat Detection Module
Implements beat detection, morphological analysis, and feature extraction
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class PPGFeatureExtractor:
    """Main class for PPG beat detection and feature extraction"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        
    def detect_beats(self, ppg_signal, min_distance=None, height_threshold=0.3):
        """
        Detect beats in PPG signal using peak detection
        
        Args:
            ppg_signal: Preprocessed PPG signal
            min_distance: Minimum distance between beats (samples)
            height_threshold: Minimum peak height (relative to signal range)
            
        Returns:
            peak_indices: Array of peak locations
            peak_properties: Dictionary of peak properties
        """
        if min_distance is None:
            # Assume minimum heart rate of 40 BPM
            min_distance = int(60 / 120 * self.fs)  # 0.5 seconds
        
        # Normalize signal for consistent thresholding
        normalized_signal = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))
        
        # Find peaks
        peaks, properties = find_peaks(
            normalized_signal,
            height=height_threshold,
            distance=min_distance,
            prominence=0.1,
            width=int(0.05 * self.fs)  # Minimum width of 50ms
        )
        
        # Additional validation of peaks
        valid_peaks = []
        valid_properties = {key: [] for key in properties.keys()}
        
        for i, peak in enumerate(peaks):
            # Check if peak is not at signal boundaries
            if peak > int(0.1 * self.fs) and peak < len(ppg_signal) - int(0.1 * self.fs):
                valid_peaks.append(peak)
                for key in properties.keys():
                    valid_properties[key].append(properties[key][i])
        
        valid_peaks = np.array(valid_peaks)
        for key in valid_properties.keys():
            valid_properties[key] = np.array(valid_properties[key])
        
        return valid_peaks, valid_properties
    
    def calculate_heart_rate(self, peak_indices):
        """
        Calculate heart rate from peak intervals
        
        Args:
            peak_indices: Array of peak locations
            
        Returns:
            hr_bpm: Heart rate in beats per minute
            rr_intervals: R-R intervals in seconds
        """
        if len(peak_indices) < 2:
            return 0, np.array([])
        
        # Calculate R-R intervals
        rr_intervals = np.diff(peak_indices) / self.fs  # Convert to seconds
        
        # Calculate heart rate
        mean_rr = np.mean(rr_intervals)
        hr_bpm = 60 / mean_rr if mean_rr > 0 else 0
        
        return hr_bpm, rr_intervals
    
    def detect_pulse_components(self, ppg_segment, peak_idx):
        """
        Detect systolic peak, dicrotic notch, and diastolic peak in a single beat
        
        Args:
            ppg_segment: PPG signal segment containing one beat
            peak_idx: Index of systolic peak within the segment
            
        Returns:
            components: Dictionary with component locations and amplitudes
        """
        components = {
            'systolic_peak': {'index': peak_idx, 'amplitude': ppg_segment[peak_idx]},
            'dicrotic_notch': {'index': None, 'amplitude': None},
            'diastolic_peak': {'index': None, 'amplitude': None}
        }
        
        # Look for dicrotic notch after systolic peak
        search_start = peak_idx + int(0.1 * self.fs)  # Start search 100ms after systolic peak
        search_end = min(len(ppg_segment), peak_idx + int(0.4 * self.fs))  # End search 400ms after
        
        if search_start < search_end:
            search_segment = ppg_segment[search_start:search_end]
            
            # Find minimum (dicrotic notch) - look for valley
            min_idx = np.argmin(search_segment)
            dicrotic_notch_idx = search_start + min_idx
            
            components['dicrotic_notch']['index'] = dicrotic_notch_idx
            components['dicrotic_notch']['amplitude'] = ppg_segment[dicrotic_notch_idx]
            
            # Look for diastolic peak after dicrotic notch
            dia_search_start = dicrotic_notch_idx + int(0.05 * self.fs)
            dia_search_end = min(len(ppg_segment), dicrotic_notch_idx + int(0.2 * self.fs))
            
            if dia_search_start < dia_search_end:
                dia_segment = ppg_segment[dia_search_start:dia_search_end]
                
                # Find local maxima for diastolic peak
                dia_peaks, _ = find_peaks(dia_segment, height=components['dicrotic_notch']['amplitude'])
                
                if len(dia_peaks) > 0:
                    # Take the first significant peak
                    diastolic_peak_idx = dia_search_start + dia_peaks[0]
                    components['diastolic_peak']['index'] = diastolic_peak_idx
                    components['diastolic_peak']['amplitude'] = ppg_segment[diastolic_peak_idx]
        
        return components
    
    def extract_morphological_features(self, ppg_signal, peak_indices):
        """
        Extract morphological features from PPG beats
        
        Args:
            ppg_signal: Preprocessed PPG signal
            peak_indices: Array of beat locations
            
        Returns:
            features: Dictionary of extracted features
        """
        features = {
            'temporal': {},
            'morphological': {},
            'statistical': {}
        }
        
        if len(peak_indices) < 2:
            return features
        
        # Temporal features
        hr_bpm, rr_intervals = self.calculate_heart_rate(peak_indices)
        features['temporal']['heart_rate'] = hr_bpm
        features['temporal']['rr_intervals'] = rr_intervals
        features['temporal']['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0
        features['temporal']['hrv_sdnn'] = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
        
        # Initialize morphological feature lists
        pulse_amplitudes = []
        pulse_widths = []
        rise_times = []
        areas_under_curve = []
        systolic_amplitudes = []
        dicrotic_notch_ratios = []
        
        # Analyze individual beats
        beat_window = int(0.8 * self.fs)  # 800ms window around each beat
        
        for i, peak in enumerate(peak_indices):
            start_idx = max(0, peak - beat_window // 2)
            end_idx = min(len(ppg_signal), peak + beat_window // 2)
            
            beat_segment = ppg_signal[start_idx:end_idx]
            local_peak_idx = peak - start_idx
            
            if local_peak_idx >= len(beat_segment):
                continue
            
            # Pulse amplitude
            baseline = np.min(beat_segment)
            amplitude = beat_segment[local_peak_idx] - baseline
            pulse_amplitudes.append(amplitude)
            systolic_amplitudes.append(beat_segment[local_peak_idx])
            
            # Pulse width at 50% amplitude
            try:
                widths, _, _, _ = peak_widths(beat_segment, [local_peak_idx], rel_height=0.5)
                pulse_widths.append(widths[0] / self.fs)  # Convert to seconds
            except:
                pulse_widths.append(0)
            
            # Rise time (time from 10% to 90% of peak amplitude)
            try:
                peak_amp = beat_segment[local_peak_idx]
                amp_10 = baseline + 0.1 * (peak_amp - baseline)
                amp_90 = baseline + 0.9 * (peak_amp - baseline)
                
                # Find indices where signal crosses these thresholds
                before_peak = beat_segment[:local_peak_idx]
                
                idx_10 = np.where(before_peak >= amp_10)[0]
                idx_90 = np.where(before_peak >= amp_90)[0]
                
                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_time = (idx_90[0] - idx_10[0]) / self.fs
                    rise_times.append(rise_time)
                else:
                    rise_times.append(0)
            except:
                rise_times.append(0)
            
            # Area under curve
            auc = np.trapz(beat_segment - baseline) / self.fs
            areas_under_curve.append(auc)
            
            # Detect pulse components
            components = self.detect_pulse_components(beat_segment, local_peak_idx)
            
            # Dicrotic notch ratio
            if components['dicrotic_notch']['amplitude'] is not None:
                dn_ratio = components['dicrotic_notch']['amplitude'] / components['systolic_peak']['amplitude']
                dicrotic_notch_ratios.append(dn_ratio)
        
        # Store morphological features
        features['morphological']['pulse_amplitudes'] = pulse_amplitudes
        features['morphological']['mean_pulse_amplitude'] = np.mean(pulse_amplitudes) if pulse_amplitudes else 0
        features['morphological']['std_pulse_amplitude'] = np.std(pulse_amplitudes) if pulse_amplitudes else 0
        
        features['morphological']['pulse_widths'] = pulse_widths
        features['morphological']['mean_pulse_width'] = np.mean(pulse_widths) if pulse_widths else 0
        
        features['morphological']['rise_times'] = rise_times
        features['morphological']['mean_rise_time'] = np.mean(rise_times) if rise_times else 0
        
        features['morphological']['areas_under_curve'] = areas_under_curve
        features['morphological']['mean_auc'] = np.mean(areas_under_curve) if areas_under_curve else 0
        
        features['morphological']['dicrotic_notch_ratios'] = dicrotic_notch_ratios
        features['morphological']['mean_dn_ratio'] = np.mean(dicrotic_notch_ratios) if dicrotic_notch_ratios else 0
        
        # Statistical features of the entire signal
        features['statistical']['signal_mean'] = np.mean(ppg_signal)
        features['statistical']['signal_std'] = np.std(ppg_signal)
        features['statistical']['signal_skewness'] = skew(ppg_signal)
        features['statistical']['signal_kurtosis'] = kurtosis(ppg_signal)
        features['statistical']['signal_range'] = np.ptp(ppg_signal)
        
        return features
    
    def extract_frequency_features(self, ppg_signal):
        """
        Extract frequency domain features from PPG signal
        
        Args:
            ppg_signal: Preprocessed PPG signal
            
        Returns:
            freq_features: Dictionary of frequency domain features
        """
        # Compute power spectral density
        freqs, psd = signal.welch(ppg_signal, fs=self.fs, nperseg=min(len(ppg_signal)//2, 512))
        
        # Define frequency bands
        vlf_band = (0.0, 0.04)    # Very low frequency
        lf_band = (0.04, 0.15)    # Low frequency
        hf_band = (0.15, 0.4)     # High frequency
        pulse_band = (0.5, 4.0)   # Main pulse frequency band
        
        def band_power(freqs, psd, band):
            """Calculate power in a specific frequency band"""
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            return np.trapz(psd[idx], freqs[idx])
        
        # Calculate band powers
        total_power = np.trapz(psd, freqs)
        vlf_power = band_power(freqs, psd, vlf_band)
        lf_power = band_power(freqs, psd, lf_band)
        hf_power = band_power(freqs, psd, hf_band)
        pulse_power = band_power(freqs, psd, pulse_band)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Calculate spectral entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        
        freq_features = {
            'total_power': total_power,
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'pulse_power': pulse_power,
            'lf_hf_ratio': lf_power / (hf_power + 1e-10),
            'dominant_frequency': dominant_freq,
            'spectral_entropy': spectral_entropy,
            'power_spectral_density': psd,
            'frequencies': freqs
        }
        
        return freq_features
    
    def extract_all_features(self, ppg_signal):
        """
        Extract all features from PPG signal
        
        Args:
            ppg_signal: Preprocessed PPG signal
            
        Returns:
            all_features: Comprehensive feature dictionary
        """
        # Detect beats
        peaks, peak_properties = self.detect_beats(ppg_signal)
        
        # Extract morphological features
        morphological_features = self.extract_morphological_features(ppg_signal, peaks)
        
        # Extract frequency features
        frequency_features = self.extract_frequency_features(ppg_signal)
        
        # Combine all features
        all_features = {
            'peaks': peaks,
            'peak_properties': peak_properties,
            'morphological': morphological_features,
            'frequency': frequency_features,
            'signal_length': len(ppg_signal),
            'sampling_rate': self.fs
        }
        
        return all_features

# Utility functions for feature extraction
def create_feature_vector(features):
    """
    Create a flat feature vector from extracted features for ML
    
    Args:
        features: Feature dictionary from extract_all_features
        
    Returns:
        feature_vector: 1D numpy array of features
        feature_names: List of feature names
    """
    feature_vector = []
    feature_names = []
    
    # Temporal features
    if 'morphological' in features and 'temporal' in features['morphological']:
        temp_features = features['morphological']['temporal']
        for key, value in temp_features.items():
            if np.isscalar(value):
                feature_vector.append(value)
                feature_names.append(f'temporal_{key}')
    
    # Morphological features
    if 'morphological' in features and 'morphological' in features['morphological']:
        morph_features = features['morphological']['morphological']
        for key, value in morph_features.items():
            if 'mean' in key or 'std' in key:
                feature_vector.append(value)
                feature_names.append(f'morphological_{key}')
    
    # Statistical features
    if 'morphological' in features and 'statistical' in features['morphological']:
        stat_features = features['morphological']['statistical']
        for key, value in stat_features.items():
            feature_vector.append(value)
            feature_names.append(f'statistical_{key}')
    
    # Frequency features
    if 'frequency' in features:
        freq_features = features['frequency']
        for key, value in freq_features.items():
            if key not in ['power_spectral_density', 'frequencies'] and np.isscalar(value):
                feature_vector.append(value)
                feature_names.append(f'frequency_{key}')
    
    return np.array(feature_vector), feature_names

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic PPG signal for testing
    def generate_realistic_ppg(duration=30, fs=125, heart_rate=75):
        """Generate more realistic PPG signal"""
        t = np.linspace(0, duration, int(duration * fs))
        
        # Cardiac frequency
        cardiac_freq = heart_rate / 60
        
        # Create realistic PPG morphology
        ppg_signal = np.zeros_like(t)
        
        for i, time_point in enumerate(t):
            # Main systolic component
            phase = 2 * np.pi * cardiac_freq * time_point
            
            # Systolic upstroke and peak
            systolic = np.exp(-((phase % (2*np.pi) - np.pi/2)**2) / 0.2)
            
            # Dicrotic notch and diastolic peak
            dicrotic_phase = (phase % (2*np.pi) - np.pi)
            if 0 < dicrotic_phase < np.pi:
                dicrotic = 0.3 * np.exp(-(dicrotic_phase - np.pi/3)**2 / 0.1)
            else:
                dicrotic = 0
            
            ppg_signal[i] = systolic + dicrotic
        
        # Add realistic noise
        noise = 0.05 * np.random.randn(len(t))
        
        return ppg_signal + noise, t
    
    # Test the feature extraction pipeline
    print("Testing PPG Feature Extraction Pipeline...")
    
    # Generate test signal
    ppg_signal, time = generate_realistic_ppg(duration=30, fs=125, heart_rate=75)
    
    # Initialize feature extractor
    feature_extractor = PPGFeatureExtractor(sampling_rate=125)
    
    # Extract all features
    all_features = feature_extractor.extract_all_features(ppg_signal)
    
    # Create feature vector
    feature_vector, feature_names = create_feature_vector(all_features)
    
    print(f"Detected {len(all_features['peaks'])} beats")
    print(f"Estimated heart rate: {all_features['morphological']['temporal']['heart_rate']:.1f} BPM")
    print(f"Mean pulse amplitude: {all_features['morphological']['morphological']['mean_pulse_amplitude']:.3f}")
    print(f"Dominant frequency: {all_features['frequency']['dominant_frequency']:.3f} Hz")
    print(f"Feature vector length: {len(feature_vector)}")
    print(f"First 10 features: {feature_names[:10]}")
    print("Feature extraction completed successfully!")