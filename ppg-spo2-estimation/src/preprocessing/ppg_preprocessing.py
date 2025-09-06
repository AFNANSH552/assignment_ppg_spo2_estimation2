"""
PPG Signal Preprocessing Module
Handles noise removal, filtering, and signal quality assessment
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

class PPGPreprocessor:
    """Main class for PPG signal preprocessing"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def bandpass_filter(self, data, low_freq=0.5, high_freq=8.0, order=4):
        """
        Apply bandpass filter to remove baseline drift and high-frequency noise
        
        Args:
            data: Raw PPG signal
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Higher cutoff frequency (Hz) 
            order: Filter order
        """
        low = low_freq / self.nyquist
        high = high_freq / self.nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    
    def remove_baseline_wander(self, data, window_size=None):
        """
        Remove baseline wander using median filtering
        
        Args:
            data: PPG signal
            window_size: Window size for median filter
        """
        if window_size is None:
            window_size = int(0.2 * self.fs)  # 200ms window
            
        # Make window_size odd
        if window_size % 2 == 0:
            window_size += 1
            
        baseline = signal.medfilt(data, kernel_size=window_size)
        corrected_signal = data - baseline
        
        return corrected_signal
    
    def calculate_sqi(self, data, window_length=None):
        """
        Calculate Signal Quality Index (SQI) based on multiple metrics
        
        Args:
            data: PPG signal segment
            window_length: Length of analysis window
            
        Returns:
            sqi_score: Quality score between 0 and 1
        """
        if window_length is None:
            window_length = int(5 * self.fs)  # 5 second windows
            
        sqi_scores = []
        
        for i in range(0, len(data) - window_length, window_length // 2):
            segment = data[i:i + window_length]
            
            # 1. SNR-based quality
            signal_power = np.var(segment)
            noise_power = np.var(np.diff(segment))  # High-frequency content as noise proxy
            snr = signal_power / (noise_power + 1e-10)
            snr_score = min(snr / 10, 1.0)  # Normalize
            
            # 2. Periodicity check
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (should show periodicity)
            peaks, _ = find_peaks(autocorr[int(0.3*self.fs):int(2*self.fs)], 
                                height=np.max(autocorr)*0.3)
            
            periodicity_score = len(peaks) / 5.0  # Normalize by expected peaks
            periodicity_score = min(periodicity_score, 1.0)
            
            # 3. Amplitude consistency
            peak_indices, _ = find_peaks(segment, distance=int(0.3*self.fs))
            if len(peak_indices) > 2:
                peak_amplitudes = segment[peak_indices]
                amplitude_cv = np.std(peak_amplitudes) / (np.mean(peak_amplitudes) + 1e-10)
                amplitude_score = max(0, 1 - amplitude_cv)
            else:
                amplitude_score = 0
            
            # Combine scores
            combined_score = (snr_score + periodicity_score + amplitude_score) / 3
            sqi_scores.append(combined_score)
        
        return np.array(sqi_scores)
    
    def detect_motion_artifacts(self, data, threshold=2.0):
        """
        Detect motion artifacts based on signal deviation
        
        Args:
            data: PPG signal
            threshold: Threshold for artifact detection (standard deviations)
            
        Returns:
            artifact_mask: Boolean mask indicating artifact locations
        """
        # Calculate moving standard deviation
        window_size = int(2 * self.fs)  # 2 second window
        moving_std = np.array([np.std(data[max(0, i-window_size//2):
                                          min(len(data), i+window_size//2)]) 
                              for i in range(len(data))])
        
        # Detect outliers
        med_std = np.median(moving_std)
        mad_std = np.median(np.abs(moving_std - med_std))
        
        artifact_mask = moving_std > (med_std + threshold * mad_std)
        
        return artifact_mask
    
    def adaptive_filter_artifacts(self, data, artifact_mask):
        """
        Apply adaptive filtering to remove motion artifacts
        
        Args:
            data: PPG signal
            artifact_mask: Boolean mask of artifact locations
            
        Returns:
            cleaned_data: Signal with artifacts removed
        """
        cleaned_data = data.copy()
        
        # Find artifact segments
        artifact_segments = []
        in_artifact = False
        start_idx = 0
        
        for i, is_artifact in enumerate(artifact_mask):
            if is_artifact and not in_artifact:
                start_idx = i
                in_artifact = True
            elif not is_artifact and in_artifact:
                artifact_segments.append((start_idx, i))
                in_artifact = False
        
        if in_artifact:  # Handle case where signal ends in artifact
            artifact_segments.append((start_idx, len(artifact_mask)))
        
        # Replace artifact segments with interpolated values
        for start, end in artifact_segments:
            if start > 0 and end < len(cleaned_data):
                # Linear interpolation
                cleaned_data[start:end] = np.linspace(cleaned_data[start-1], 
                                                     cleaned_data[end], 
                                                     end-start)
            elif start == 0:
                cleaned_data[start:end] = cleaned_data[end]
            else:
                cleaned_data[start:end] = cleaned_data[start-1]
        
        return cleaned_data
    
    def preprocess_pipeline(self, raw_data, return_quality=True):
        """
        Complete preprocessing pipeline
        
        Args:
            raw_data: Raw PPG signal
            return_quality: Whether to return quality metrics
            
        Returns:
            processed_data: Cleaned PPG signal
            quality_metrics: Dictionary of quality metrics (if requested)
        """
        # Step 1: Bandpass filtering
        filtered_data = self.bandpass_filter(raw_data)
        
        # Step 2: Baseline correction
        baseline_corrected = self.remove_baseline_wander(filtered_data)
        
        # Step 3: Motion artifact detection and removal
        artifact_mask = self.detect_motion_artifacts(baseline_corrected)
        cleaned_data = self.adaptive_filter_artifacts(baseline_corrected, artifact_mask)
        
        # Step 4: Final filtering
        final_data = self.bandpass_filter(cleaned_data)
        
        if return_quality:
            # Calculate quality metrics
            sqi_scores = self.calculate_sqi(final_data)
            artifact_percentage = np.sum(artifact_mask) / len(artifact_mask) * 100
            
            quality_metrics = {
                'sqi_scores': sqi_scores,
                'mean_sqi': np.mean(sqi_scores),
                'artifact_percentage': artifact_percentage,
                'signal_length': len(final_data),
                'sampling_rate': self.fs
            }
            
            return final_data, quality_metrics
        
        return final_data

# Utility functions for preprocessing
def normalize_signal(data):
    """Normalize signal to zero mean and unit variance"""
    return (data - np.mean(data)) / np.std(data)

def segment_signal(data, segment_length, overlap=0.5):
    """
    Segment signal into overlapping windows
    
    Args:
        data: Input signal
        segment_length: Length of each segment
        overlap: Overlap fraction (0-1)
        
    Returns:
        segments: List of signal segments
    """
    step_size = int(segment_length * (1 - overlap))
    segments = []
    
    for i in range(0, len(data) - segment_length, step_size):
        segments.append(data[i:i + segment_length])
    
    return segments

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic PPG signal for testing
    def generate_synthetic_ppg(duration=30, fs=125):
        """Generate synthetic PPG signal with noise and artifacts"""
        t = np.linspace(0, duration, int(duration * fs))
        
        # Base PPG signal (simplified cardiac cycle)
        heart_rate = 75  # BPM
        cardiac_freq = heart_rate / 60
        
        # Main pulse component
        ppg_signal = np.sin(2 * np.pi * cardiac_freq * t)
        
        # Add harmonics for realistic shape
        ppg_signal += 0.3 * np.sin(4 * np.pi * cardiac_freq * t)
        ppg_signal += 0.1 * np.sin(6 * np.pi * cardiac_freq * t)
        
        # Add baseline wander
        baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        
        # Add noise
        noise = 0.1 * np.random.randn(len(t))
        
        # Add motion artifacts (random spikes)
        artifact_indices = np.random.choice(len(t), size=int(0.05 * len(t)), replace=False)
        artifacts = np.zeros_like(t)
        artifacts[artifact_indices] = np.random.randn(len(artifact_indices)) * 0.8
        
        return ppg_signal + baseline + noise + artifacts, t
    
    # Test the preprocessing pipeline
    print("Testing PPG Preprocessing Pipeline...")
    
    # Generate test signal
    raw_ppg, time = generate_synthetic_ppg(duration=30, fs=125)
    
    # Initialize preprocessor
    preprocessor = PPGPreprocessor(sampling_rate=125)
    
    # Run preprocessing pipeline
    processed_ppg, quality_metrics = preprocessor.preprocess_pipeline(raw_ppg)
    
    print(f"Signal length: {len(raw_ppg)} samples")
    print(f"Mean SQI: {quality_metrics['mean_sqi']:.3f}")
    print(f"Artifact percentage: {quality_metrics['artifact_percentage']:.1f}%")
    print("Preprocessing completed successfully!")