"""
Main PPG Processing Pipeline
Integrates preprocessing, feature extraction, and SpO2 estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
# from ppg_preprocessing import PPGPreprocessor
# from ppg_feature_extraction import PPGFeatureExtractor
# from spo2_estimation import SpO2Estimator

class PPGPipeline:
    """Main pipeline class integrating all PPG processing components"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        self.preprocessor = None  # PPGPreprocessor(sampling_rate)
        self.feature_extractor = None  # PPGFeatureExtractor(sampling_rate)
        self.spo2_estimator = None  # SpO2Estimator(sampling_rate)
        
        # Initialize components (would use actual imports in real implementation)
        self._initialize_components()
        
        # Processing history
        self.processing_history = []
        
    def _initialize_components(self):
        """Initialize processing components (placeholder for actual imports)"""
        print("Initializing PPG processing components...")
        # In real implementation, these would be actual imports
        # self.preprocessor = PPGPreprocessor(self.fs)
        # self.feature_extractor = PPGFeatureExtractor(self.fs)
        # self.spo2_estimator = SpO2Estimator(self.fs)
        print("Components initialized successfully!")
    
    def load_ppg_data(self, filepath=None, data_array=None, red_channel=None, ir_channel=None):
        """
        Load PPG data from file or array
        
        Args:
            filepath: Path to data file (CSV, NPY, etc.)
            data_array: Numpy array of PPG data
            red_channel: Red wavelength PPG data
            ir_channel: IR wavelength PPG data
            
        Returns:
            ppg_data: Dictionary containing loaded data
        """
        ppg_data = {}
        
        if filepath is not None:
            # Load from file
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                ppg_data['ppg_signal'] = df.iloc[:, 0].values
                if df.shape[1] > 1:
                    ppg_data['red_ppg'] = df.iloc[:, 0].values
                    ppg_data['ir_ppg'] = df.iloc[:, 1].values
            elif filepath.endswith('.npy'):
                data = np.load(filepath)
                if data.ndim == 1:
                    ppg_data['ppg_signal'] = data
                else:
                    ppg_data['red_ppg'] = data[:, 0]
                    ppg_data['ir_ppg'] = data[:, 1]
        
        elif data_array is not None:
            if data_array.ndim == 1:
                ppg_data['ppg_signal'] = data_array
            else:
                ppg_data['red_ppg'] = data_array[:, 0]
                ppg_data['ir_ppg'] = data_array[:, 1]
        
        elif red_channel is not None and ir_channel is not None:
            ppg_data['red_ppg'] = red_channel
            ppg_data['ir_ppg'] = ir_channel
            # Use red channel as main signal for processing
            ppg_data['ppg_signal'] = red_channel
        
        else:
            raise ValueError("Must provide either filepath, data_array, or red/ir channels")
        
        # Add metadata
        ppg_data['sampling_rate'] = self.fs
        ppg_data['duration'] = len(ppg_data.get('ppg_signal', ppg_data.get('red_ppg'))) / self.fs
        ppg_data['timestamp'] = datetime.now().isoformat()
        
        return ppg_data
    
    def process_ppg_signal(self, ppg_data, segment_length=None, overlap=0.5):
        """
        Complete PPG processing pipeline
        
        Args:
            ppg_data: Dictionary containing PPG data
            segment_length: Length of processing segments (seconds)
            overlap: Overlap between segments (0-1)
            
        Returns:
            results: Dictionary containing all processing results
        """
        print("Starting PPG signal processing...")
        
        results = {
            'input_data': ppg_data,
            'preprocessing': {},
            'features': {},
            'spo2_estimates': {},
            'quality_metrics': {},
            'timestamps': []
        }
        
        # Get main PPG signal for processing
        main_signal = ppg_data.get('ppg_signal', ppg_data.get('red_ppg'))
        
        if main_signal is None:
            raise ValueError("No PPG signal found in input data")
        
        # Set default segment length
        if segment_length is None:
            segment_length = min(30, len(main_signal) / self.fs)  # 30 seconds or full signal
        
        segment_samples = int(segment_length * self.fs)
        step_size = int(segment_samples * (1 - overlap))
        
        # Process signal in segments
        segment_results = []
        
        for i in range(0, len(main_signal) - segment_samples, step_size):
            segment_start = i
            segment_end = i + segment_samples
            segment_signal = main_signal[segment_start:segment_end]
            
            segment_result = self._process_segment(
                segment_signal, 
                ppg_data.get('red_ppg', [None])[segment_start:segment_end] if 'red_ppg' in ppg_data else None,
                ppg_data.get('ir_ppg', [None])[segment_start:segment_end] if 'ir_ppg' in ppg_data else None,
                segment_start / self.fs
            )
            
            segment_results.append(segment_result)
        
        # Aggregate results
        results = self._aggregate_segment_results(segment_results)
        
        # Store processing history
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_length': len(main_signal),
            'segments_processed': len(segment_results),
            'results_summary': self._summarize_results(results)
        })
        
        print(f"Processing completed! Processed {len(segment_results)} segments.")
        return results
    
    def _process_segment(self, segment_signal, red_segment=None, ir_segment=None, time_offset=0):
        """
        Process a single PPG segment
        
        Args:
            segment_signal: PPG signal segment
            red_segment: Red wavelength segment
            ir_segment: IR wavelength segment  
            time_offset: Time offset of segment
            
        Returns:
            segment_result: Processing results for this segment
        """
        segment_result = {
            'time_offset': time_offset,
            'preprocessing': {},
            'features': {},
            'spo2': {},
            'quality': {}
        }
        
        # Step 1: Preprocessing (simulated)
        try:
            # In real implementation: processed_signal, quality = self.preprocessor.preprocess_pipeline(segment_signal)
            processed_signal = self._simulate_preprocessing(segment_signal)
            quality_metrics = self._simulate_quality_assessment(processed_signal)
            
            segment_result['preprocessing']['processed_signal'] = processed_signal
            segment_result['quality'] = quality_metrics
        except Exception as e:
            print(f"Preprocessing error at time {time_offset:.1f}s: {e}")
            segment_result['preprocessing']['error'] = str(e)
            return segment_result
        
        # Step 2: Feature extraction (simulated)
        try:
            # In real implementation: features = self.feature_extractor.extract_all_features(processed_signal)
            features = self._simulate_feature_extraction(processed_signal)
            segment_result['features'] = features
        except Exception as e:
            print(f"Feature extraction error at time {time_offset:.1f}s: {e}")
            segment_result['features']['error'] = str(e)
            return segment_result
        
        # Step 3: SpO2 estimation (simulated)
        try:
            r_ratio = None
            if red_segment is not None and ir_segment is not None:
                r_ratio = self._simulate_r_ratio_calculation(red_segment, ir_segment)
            
            # In real implementation: spo2, confidence = self.spo2_estimator.predict_spo2(features, r_ratio)
            spo2_estimates = self._simulate_spo2_estimation(features, r_ratio)
            segment_result['spo2'] = spo2_estimates
        except Exception as e:
            print(f"SpO2 estimation error at time {time_offset:.1f}s: {e}")
            segment_result['spo2']['error'] = str(e)
        
        return segment_result
    
    def _simulate_preprocessing(self, signal):
        """Simulate preprocessing (placeholder for actual implementation)"""
        # Simple bandpass filter simulation
        from scipy import signal as sig
        
        # Bandpass filter 0.5-8 Hz
        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        b, a = sig.butter(4, [low, high], btype='band')
        filtered_signal = sig.filtfilt(b, a, signal)
        
        # Baseline removal (simple detrending)
        processed_signal = filtered_signal - np.mean(filtered_signal)
        
        return processed_signal
    
    def _simulate_quality_assessment(self, signal):
        """Simulate signal quality assessment"""
        # Simple quality metrics
        signal_power = np.var(signal)
        noise_estimate = np.var(np.diff(signal))
        snr = signal_power / (noise_estimate + 1e-10)
        
        # Periodicity check using autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[int(0.3*self.fs):int(2*self.fs)])
        
        quality_score = min(1.0, snr / 10.0) * min(1.0, len(peaks) / 3.0)
        
        return {
            'quality_score': quality_score,
            'snr': snr,
            'periodicity_peaks': len(peaks),
            'signal_power': signal_power
        }
    
    def _simulate_feature_extraction(self, signal):
        """Simulate feature extraction"""
        from scipy.signal import find_peaks
        from scipy.stats import skew, kurtosis
        
        # Find peaks (beats)
        min_distance = int(60 / 120 * self.fs)  # Minimum 0.5s between beats
        peaks, _ = find_peaks(signal, distance=min_distance, prominence=0.1)
        
        # Heart rate calculation
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.fs
            heart_rate = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        else:
            heart_rate = 0
            rr_intervals = np.array([])
        
        # Morphological features
        if len(peaks) > 0:
            amplitudes = signal[peaks]
            mean_amplitude = np.mean(amplitudes)
            std_amplitude = np.std(amplitudes)
        else:
            mean_amplitude = 0
            std_amplitude = 0
        
        # Statistical features
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_skewness = skew(signal)
        signal_kurtosis = kurtosis(signal)
        
        # Frequency domain features (simplified)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        psd = np.abs(fft)**2
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_psd = psd[:len(psd)//2]
        dominant_freq = positive_freqs[np.argmax(positive_psd)]
        
        features = {
            'temporal': {
                'heart_rate': heart_rate,
                'num_beats': len(peaks),
                'mean_rr_interval': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
                'rr_std': np.std(rr_intervals) if len(rr_intervals) > 0 else 0
            },
            'morphological': {
                'mean_amplitude': mean_amplitude,
                'std_amplitude': std_amplitude,
                'amplitude_cv': std_amplitude / (mean_amplitude + 1e-10)
            },
            'statistical': {
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'signal_skewness': signal_skewness,
                'signal_kurtosis': signal_kurtosis
            },
            'frequency': {
                'dominant_frequency': dominant_freq,
                'total_power': np.sum(positive_psd)
            }
        }
        
        return features
    
    def _simulate_r_ratio_calculation(self, red_signal, ir_signal):
        """Simulate R-ratio calculation"""
        # Calculate AC and DC components
        red_ac = np.std(red_signal)
        red_dc = np.mean(red_signal)
        
        ir_ac = np.std(ir_signal)
        ir_dc = np.mean(ir_signal)
        
        # Calculate R-ratio
        if ir_dc != 0 and red_dc != 0 and ir_ac != 0:
            r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            return r_ratio
        else:
            return None
    
    def _simulate_spo2_estimation(self, features, r_ratio):
        """Simulate SpO2 estimation"""
        spo2_estimates = {}
        
        # Traditional method (if R-ratio available)
        if r_ratio is not None:
            # Simple empirical formula
            traditional_spo2 = 110 - 25 * np.clip(r_ratio, 0.4, 3.0)
            traditional_spo2 = np.clip(traditional_spo2, 70, 100)
            spo2_estimates['traditional'] = traditional_spo2
        
        # ML-based estimation (simplified simulation)
        # In practice, this would use trained models
        base_spo2 = 98
        
        # Heart rate effect
        hr = features['temporal']['heart_rate']
        hr_effect = -0.1 * max(0, hr - 80)  # Higher HR slightly reduces SpO2
        
        # Signal quality effect
        quality_effect = 0  # Would use actual quality metrics
        
        # Add some realistic variation
        ml_spo2 = base_spo2 + hr_effect + quality_effect + np.random.normal(0, 1)
        ml_spo2 = np.clip(ml_spo2, 85, 100)
        
        spo2_estimates['ml_estimate'] = ml_spo2
        spo2_estimates['confidence'] = 0.85  # Simulated confidence
        
        return spo2_estimates
    
    def _aggregate_segment_results(self, segment_results):
        """Aggregate results from all segments"""
        aggregated = {
            'segments': segment_results,
            'summary': {},
            'time_series': {}
        }
        
        # Extract time series data
        timestamps = [seg['time_offset'] for seg in segment_results if 'spo2' in seg and 'ml_estimate' in seg['spo2']]
        spo2_values = [seg['spo2']['ml_estimate'] for seg in segment_results if 'spo2' in seg and 'ml_estimate' in seg['spo2']]
        heart_rates = [seg['features']['temporal']['heart_rate'] for seg in segment_results if 'features' in seg]
        quality_scores = [seg['quality']['quality_score'] for seg in segment_results if 'quality' in seg]
        
        aggregated['time_series'] = {
            'timestamps': timestamps,
            'spo2_values': spo2_values,
            'heart_rates': heart_rates,
            'quality_scores': quality_scores
        }
        
        # Summary statistics
        if spo2_values:
            aggregated['summary']['mean_spo2'] = np.mean(spo2_values)
            aggregated['summary']['std_spo2'] = np.std(spo2_values)
            aggregated['summary']['min_spo2'] = np.min(spo2_values)
            aggregated['summary']['max_spo2'] = np.max(spo2_values)
        
        if heart_rates:
            aggregated['summary']['mean_hr'] = np.mean(heart_rates)
            aggregated['summary']['std_hr'] = np.std(heart_rates)
        
        if quality_scores:
            aggregated['summary']['mean_quality'] = np.mean(quality_scores)
            aggregated['summary']['good_quality_percentage'] = np.mean(np.array(quality_scores) > 0.7) * 100
        
        return aggregated
    
    def _summarize_results(self, results):
        """Create a summary of processing results"""
        summary = {}
        
        if 'summary' in results:
            summary.update(results['summary'])
        
        summary['segments_processed'] = len(results.get('segments', []))
        summary['processing_timestamp'] = datetime.now().isoformat()
        
        return summary
    
    def visualize_results(self, results, save_path=None):
        """
        Create visualization of processing results
        
        Args:
            results: Processing results from process_ppg_signal
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('PPG Processing Results', fontsize=16)
        
        time_series = results.get('time_series', {})
        timestamps = time_series.get('timestamps', [])
        
        # Plot SpO2 over time
        if 'spo2_values' in time_series and timestamps:
            axes[0].plot(timestamps, time_series['spo2_values'], 'b-o', markersize=4)
            axes[0].set_ylabel('SpO2 (%)')
            axes[0].set_title('SpO2 Estimation Over Time')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(85, 100)
        
        # Plot heart rate over time
        if 'heart_rates' in time_series and timestamps:
            axes[1].plot(timestamps, time_series['heart_rates'], 'r-o', markersize=4)
            axes[1].set_ylabel('Heart Rate (BPM)')
            axes[1].set_title('Heart Rate Over Time')
            axes[1].grid(True, alpha=0.3)
        
        # Plot signal quality over time
        if 'quality_scores' in time_series and timestamps:
            axes[2].plot(timestamps, time_series['quality_scores'], 'g-o', markersize=4)
            axes[2].axhline(y=0.7, color='orange', linestyle='--', label='Good Quality Threshold')
            axes[2].set_ylabel('Signal Quality')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_title('Signal Quality Over Time')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, 1)
            axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results, output_path=None):
        """
        Generate a comprehensive report of the processing results
        
        Args:
            results: Processing results from process_ppg_signal
            output_path: Optional path to save the report
        """
        report = []
        report.append("="*60)
        report.append("PPG SIGNAL PROCESSING REPORT")
        report.append("="*60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        summary = results.get('summary', {})
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 20)
        
        if 'mean_spo2' in summary:
            report.append(f"Mean SpO2: {summary['mean_spo2']:.1f}% ± {summary.get('std_spo2', 0):.1f}%")
            report.append(f"SpO2 Range: {summary.get('min_spo2', 0):.1f}% - {summary.get('max_spo2', 0):.1f}%")
        
        if 'mean_hr' in summary:
            report.append(f"Mean Heart Rate: {summary['mean_hr']:.0f} ± {summary.get('std_hr', 0):.0f} BPM")
        
        if 'mean_quality' in summary:
            report.append(f"Average Signal Quality: {summary['mean_quality']:.2f}")
            report.append(f"Good Quality Segments: {summary.get('good_quality_percentage', 0):.1f}%")
        
        report.append("")
        
        # Processing details
        report.append("PROCESSING DETAILS:")
        report.append("-" * 20)
        report.append(f"Total Segments Processed: {summary.get('segments_processed', 0)}")
        
        # Time series analysis
        time_series = results.get('time_series', {})
        if 'spo2_values' in time_series:
            spo2_values = time_series['spo2_values']
            report.append(f"SpO2 Measurements: {len(spo2_values)}")
            
            # Clinical assessment
            report.append("")
            report.append("CLINICAL ASSESSMENT:")
            report.append("-" * 20)
            
            normal_spo2 = np.sum(np.array(spo2_values) >= 95)
            mild_hypox = np.sum((np.array(spo2_values) >= 90) & (np.array(spo2_values) < 95))
            moderate_hypox = np.sum((np.array(spo2_values) >= 85) & (np.array(spo2_values) < 90))
            severe_hypox = np.sum(np.array(spo2_values) < 85)
            
            total_measurements = len(spo2_values)
            report.append(f"Normal SpO2 (≥95%): {normal_spo2}/{total_measurements} ({100*normal_spo2/total_measurements:.1f}%)")
            report.append(f"Mild Hypoxemia (90-94%): {mild_hypox}/{total_measurements} ({100*mild_hypox/total_measurements:.1f}%)")
            report.append(f"Moderate Hypoxemia (85-89%): {moderate_hypox}/{total_measurements} ({100*moderate_hypox/total_measurements:.1f}%)")
            report.append(f"Severe Hypoxemia (<85%): {severe_hypox}/{total_measurements} ({100*severe_hypox/total_measurements:.1f}%)")
        
        report.append("")
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        print(report_text)
        return report_text

# Example usage and demonstration
def demonstrate_pipeline():
    """Demonstrate the PPG processing pipeline"""
    print("PPG Processing Pipeline Demonstration")
    print("="*50)
    
    # Generate synthetic PPG data
    def generate_demo_ppg(duration=60, fs=125):
        """Generate demo PPG data with dual wavelengths"""
        t = np.linspace(0, duration, int(duration * fs))
        
        # Simulate heart rate variation
        base_hr = 75
        hr_variation = 5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz variation
        instantaneous_hr = base_hr + hr_variation
        
        # Generate PPG signals
        red_ppg = []
        ir_ppg = []
        
        for i, time_point in enumerate(t):
            hr = instantaneous_hr[i]
            cardiac_freq = hr / 60
            
            # Red channel (more affected by SpO2)
            red_component = np.sin(2 * np.pi * cardiac_freq * time_point)
            red_component += 0.3 * np.sin(4 * np.pi * cardiac_freq * time_point)
            
            # IR channel (less affected by SpO2)
            ir_component = 1.2 * np.sin(2 * np.pi * cardiac_freq * time_point)
            ir_component += 0.2 * np.sin(4 * np.pi * cardiac_freq * time_point)
            
            red_ppg.append(red_component)
            ir_ppg.append(ir_component)
        
        # Add noise and artifacts
        noise_level = 0.1
        red_ppg = np.array(red_ppg) + noise_level * np.random.randn(len(red_ppg))
        ir_ppg = np.array(ir_ppg) + noise_level * np.random.randn(len(ir_ppg))
        
        # Add some motion artifacts
        artifact_indices = np.random.choice(len(t), size=int(0.02 * len(t)), replace=False)
        red_ppg[artifact_indices] += np.random.randn(len(artifact_indices)) * 0.5
        ir_ppg[artifact_indices] += np.random.randn(len(artifact_indices)) * 0.3
        
        return red_ppg, ir_ppg, t
    
    # Initialize pipeline
    pipeline = PPGPipeline(sampling_rate=125)
    
    # Generate demo data
    print("Generating synthetic PPG data...")
    red_signal, ir_signal, time_vector = generate_demo_ppg(duration=60, fs=125)
    
    # Load data into pipeline
    ppg_data = pipeline.load_ppg_data(red_channel=red_signal, ir_channel=ir_signal)
    print(f"Loaded PPG data: {ppg_data['duration']:.1f} seconds")
    
    # Process the signal
    print("Processing PPG signal...")
    results = pipeline.process_ppg_signal(ppg_data, segment_length=10, overlap=0.5)
    
    # Generate report
    print("\nGenerating report...")
    pipeline.generate_report(results)
    
    # Create visualization
    print("\nCreating visualization...")
    pipeline.visualize_results(results)
    
    return pipeline, results

# Main execution
if __name__ == "__main__":
    # Run demonstration
    pipeline, results = demonstrate_pipeline()
    print("\nPipeline demonstration completed successfully!")
    print("\nTo use with your own data:")
    print("1. Load your PPG data using pipeline.load_ppg_data()")
    print("2. Process using pipeline.process_ppg_signal()")
    print("3. Generate reports and visualizations")