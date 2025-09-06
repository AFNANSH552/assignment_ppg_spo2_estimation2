"""
PPG Demo Application with Real-time Processing
Interactive demo for PPG signal processing and SpO2 estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from datetime import datetime
import json

class PPGRealtimeDemo:
    """Real-time PPG processing demo application"""
    
    def __init__(self):
        self.fs = 125  # Sampling rate
        self.buffer_size = 10 * self.fs  # 10-second buffer
        self.processing_window = 5 * self.fs  # 5-second processing window
        
        # Data buffers
        self.data_buffer = queue.Queue(maxsize=self.buffer_size)
        self.results_buffer = queue.Queue(maxsize=100)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        
        # Results storage
        self.timestamps = []
        self.spo2_values = []
        self.heart_rates = []
        self.quality_scores = []
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("PPG Real-time Processing Demo")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File input
        ttk.Label(control_frame, text="Data Input:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(control_frame, textvariable=self.file_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(control_frame, text="Load PPG File", 
                  command=self.load_file).grid(row=0, column=2, padx=(10, 0))
        
        ttk.Button(control_frame, text="Generate Demo Data", 
                  command=self.generate_demo_data).grid(row=0, column=3, padx=(10, 0))
        
        # Processing controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=1, columnspan=4, sticky=tk.EW, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", 
                                     command=self.start_processing, state=tk.DISABLED)
        self.start_button.grid(row=2, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, padx=(0, 10))
        
        ttk.Button(control_frame, text="Save Results", 
                  command=self.save_results).grid(row=2, column=2, padx=(0, 10))
        
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_results).grid(row=2, column=3, padx=(0, 10))
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current readings
        readings_frame = ttk.Frame(status_frame)
        readings_frame.pack(fill=tk.X)
        
        ttk.Label(readings_frame, text="Current SpO2:").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.spo2_label = ttk.Label(readings_frame, text="--", font=('Arial', 12, 'bold'))
        self.spo2_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(readings_frame, text="Heart Rate:").grid(row=0, column=2, sticky=tk.W, padx=(20, 20))
        self.hr_label = ttk.Label(readings_frame, text="--", font=('Arial', 12, 'bold'))
        self.hr_label.grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(readings_frame, text="Signal Quality:").grid(row=0, column=4, sticky=tk.W, padx=(20, 20))
        self.quality_label = ttk.Label(readings_frame, text="--", font=('Arial', 12, 'bold'))
        self.quality_label.grid(row=0, column=5, sticky=tk.W)
        
        # Processing info
        ttk.Separator(status_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack()
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Processing Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup matplotlib figure
        self.setup_plots(results_frame)
        
        # Start GUI update timer
        self.root.after(100, self.update_gui)
        
    def setup_plots(self, parent_frame):
        """Setup matplotlib plots for real-time display"""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create figure with subplots
            self.fig = Figure(figsize=(12, 8))
            
            # SpO2 plot
            self.ax1 = self.fig.add_subplot(3, 1, 1)
            self.ax1.set_title('SpO2 Over Time')
            self.ax1.set_ylabel('SpO2 (%)')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_ylim(85, 100)
            
            # Heart rate plot
            self.ax2 = self.fig.add_subplot(3, 1, 2)
            self.ax2.set_title('Heart Rate Over Time')
            self.ax2.set_ylabel('Heart Rate (BPM)')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(40, 120)
            
            # Signal quality plot
            self.ax3 = self.fig.add_subplot(3, 1, 3)
            self.ax3.set_title('Signal Quality Over Time')
            self.ax3.set_ylabel('Quality Score')
            self.ax3.set_xlabel('Time (seconds)')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_ylim(0, 1)
            
            # Add quality threshold line
            self.ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good Quality Threshold')
            self.ax3.legend()
            
            self.fig.tight_layout()
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initialize empty line plots
            self.spo2_line, = self.ax1.plot([], [], 'b-o', markersize=4)
            self.hr_line, = self.ax2.plot([], [], 'r-o', markersize=4)
            self.quality_line, = self.ax3.plot([], [], 'g-o', markersize=4)
            
        except ImportError:
            ttk.Label(parent_frame, text="Matplotlib not available. Install matplotlib to see plots.").pack()
    
    def load_file(self):
        """Load PPG data from file"""
        file_path = filedialog.askopenfilename(
            title="Select PPG Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("NumPy files", "*.npy"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    import pandas as pd
                    data = pd.read_csv(file_path).values
                elif file_path.endswith('.npy'):
                    data = np.load(file_path)
                else:
                    data = np.loadtxt(file_path)
                
                # Store data for processing
                if data.ndim == 1:
                    self.ppg_data = {'ppg_signal': data}
                else:
                    self.ppg_data = {
                        'red_ppg': data[:, 0],
                        'ir_ppg': data[:, 1] if data.shape[1] > 1 else data[:, 0],
                        'ppg_signal': data[:, 0]
                    }
                
                self.file_var.set(f"Loaded: {file_path.split('/')[-1]} ({len(data)} samples)")
                self.start_button.config(state=tk.NORMAL)
                self.status_var.set(f"File loaded successfully. Duration: {len(data)/self.fs:.1f} seconds")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                self.file_var.set("Error loading file")
    
    def generate_demo_data(self):
        """Generate synthetic PPG data for demonstration"""
        duration = 120  # 2 minutes
        t = np.linspace(0, duration, int(duration * self.fs))
        
        # Generate realistic PPG with varying SpO2 and HR
        base_hr = 75
        hr_variation = 10 * np.sin(2 * np.pi * 0.05 * t)  # Slow HR variation
        instantaneous_hr = base_hr + hr_variation
        
        # SpO2 variation (simulate some desaturation events)
        base_spo2 = 98
        spo2_events = np.zeros_like(t)
        
        # Add some desaturation events
        event_times = [30, 60, 90]  # seconds
        for event_time in event_times:
            event_idx = int(event_time * self.fs)
            event_duration = int(10 * self.fs)  # 10 second events
            if event_idx + event_duration < len(t):
                spo2_events[event_idx:event_idx + event_duration] = -8 * np.exp(-((np.arange(event_duration) - event_duration/2)**2) / (event_duration/6)**2)
        
        instantaneous_spo2 = base_spo2 + spo2_events
        
        # Generate PPG signals based on physiological parameters
        red_ppg = []
        ir_ppg = []
        
        for i, time_point in enumerate(t):
            hr = instantaneous_hr[i]
            spo2 = instantaneous_spo2[i]
            cardiac_freq = hr / 60
            
            # Red channel (affected by SpO2)
            red_amplitude = 1.0 - (100 - spo2) * 0.02  # Lower SpO2 = higher red absorption
            red_signal = red_amplitude * np.sin(2 * np.pi * cardiac_freq * time_point)
            red_signal += 0.3 * red_amplitude * np.sin(4 * np.pi * cardiac_freq * time_point)
            
            # IR channel (less affected by SpO2)
            ir_amplitude = 1.2
            ir_signal = ir_amplitude * np.sin(2 * np.pi * cardiac_freq * time_point)
            ir_signal += 0.2 * ir_amplitude * np.sin(4 * np.pi * cardiac_freq * time_point)
            
            red_ppg.append(red_signal)
            ir_ppg.append(ir_signal)
        
        # Add realistic noise
        noise_level = 0.05
        red_ppg = np.array(red_ppg) + noise_level * np.random.randn(len(red_ppg))
        ir_ppg = np.array(ir_ppg) + noise_level * np.random.randn(len(ir_ppg))
        
        # Add some motion artifacts
        artifact_indices = np.random.choice(len(t), size=int(0.01 * len(t)), replace=False)
        red_ppg[artifact_indices] += np.random.randn(len(artifact_indices)) * 0.3
        ir_ppg[artifact_indices] += np.random.randn(len(artifact_indices)) * 0.2
        
        # Store generated data
        self.ppg_data = {
            'red_ppg': red_ppg,
            'ir_ppg': ir_ppg,
            'ppg_signal': red_ppg,
            'true_spo2': instantaneous_spo2,  # Ground truth for evaluation
            'true_hr': instantaneous_hr
        }
        
        self.file_var.set(f"Demo data generated ({duration} seconds)")
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Demo data ready for processing")
    
    def start_processing(self):
        """Start real-time PPG processing"""
        if not hasattr(self, 'ppg_data'):
            messagebox.showerror("Error", "No data loaded. Please load a file or generate demo data.")
            return
        
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Clear previous results
        self.clear_results()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.status_var.set("Processing started...")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Processing stopped")
    
    def processing_loop(self):
        """Main processing loop running in separate thread"""
        signal_data = self.ppg_data['ppg_signal']
        red_data = self.ppg_data.get('red_ppg', signal_data)
        ir_data = self.ppg_data.get('ir_ppg', signal_data)
        
        # Processing parameters
        window_size = 5 * self.fs  # 5-second windows
        step_size = int(1 * self.fs)  # 1-second steps
        
        current_pos = 0
        
        while self.is_processing and current_pos + window_size < len(signal_data):
            # Extract current window
            window_signal = signal_data[current_pos:current_pos + window_size]
            window_red = red_data[current_pos:current_pos + window_size]
            window_ir = ir_data[current_pos:current_pos + window_size]
            
            # Process the window
            results = self.process_window(window_signal, window_red, window_ir)
            
            # Add timestamp
            timestamp = current_pos / self.fs
            results['timestamp'] = timestamp
            
            # Store results
            self.results_buffer.put(results)
            
            # Move to next window
            current_pos += step_size
            
            # Simulate real-time processing delay
            time.sleep(0.5)  # Process every 0.5 seconds
        
        if self.is_processing:
            self.is_processing = False
            # Update GUI in main thread
            self.root.after_idle(lambda: self.status_var.set("Processing completed"))
    
    def process_window(self, signal, red_signal, ir_signal):
        """Process a single window of PPG data"""
        results = {}
        
        # Simple preprocessing (bandpass filter)
        filtered_signal = self.simple_bandpass_filter(signal)
        
        # Beat detection and heart rate
        heart_rate, quality = self.estimate_heart_rate(filtered_signal)
        results['heart_rate'] = heart_rate
        results['quality'] = quality
        
        # SpO2 estimation
        r_ratio = self.calculate_r_ratio(red_signal, ir_signal)
        spo2 = self.estimate_spo2(r_ratio, heart_rate)
        results['spo2'] = spo2
        results['r_ratio'] = r_ratio
        
        return results
    
    def simple_bandpass_filter(self, signal):
        """Simple bandpass filter implementation"""
        from scipy import signal as sig
        
        # Bandpass filter 0.5-8 Hz
        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        
        try:
            b, a = sig.butter(4, [low, high], btype='band')
            filtered_signal = sig.filtfilt(b, a, signal)
            return filtered_signal
        except:
            # Fallback to simple detrending if scipy not available
            return signal - np.mean(signal)
    
    def estimate_heart_rate(self, signal):
        """Estimate heart rate from PPG signal"""
        try:
            from scipy.signal import find_peaks
            
            # Normalize signal
            normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
            
            # Find peaks
            min_distance = int(60 / 120 * self.fs)  # Maximum 120 BPM
            peaks, properties = find_peaks(
                normalized,
                height=0.3,
                distance=min_distance,
                prominence=0.1
            )
            
            # Calculate heart rate
            if len(peaks) > 1:
                intervals = np.diff(peaks) / self.fs
                heart_rate = 60 / np.mean(intervals)
                
                # Quality assessment based on interval consistency
                interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                quality = max(0.1, 1.0 - interval_cv)
                
                return min(150, max(40, heart_rate)), min(1.0, quality)
            else:
                return 0, 0.1
                
        except ImportError:
            # Simple fallback method using autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak in reasonable HR range
            min_samples = int(60/150 * self.fs)  # 150 BPM max
            max_samples = int(60/40 * self.fs)   # 40 BPM min
            
            search_range = autocorr[min_samples:max_samples]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_samples
                heart_rate = 60 / (peak_idx / self.fs)
                quality = 0.7  # Moderate quality for this method
                return heart_rate, quality
            else:
                return 75, 0.3  # Default values
    
    def calculate_r_ratio(self, red_signal, ir_signal):
        """Calculate R-ratio from dual-wavelength PPG"""
        # AC and DC components
        red_ac = np.std(red_signal)
        red_dc = np.mean(red_signal)
        
        ir_ac = np.std(ir_signal)
        ir_dc = np.mean(ir_signal)
        
        # Calculate R-ratio
        if ir_dc != 0 and red_dc != 0 and ir_ac != 0:
            r_ratio = (red_ac / abs(red_dc)) / (ir_ac / abs(ir_dc))
            return max(0.1, min(3.0, r_ratio))  # Clamp to reasonable range
        else:
            return 0.8  # Default value
    
    def estimate_spo2(self, r_ratio, heart_rate):
        """Estimate SpO2 from R-ratio and other parameters"""
        # Traditional empirical formula with modifications
        base_spo2 = 110 - 25 * r_ratio
        
        # Heart rate adjustment (high HR might indicate stress/lower SpO2)
        hr_adjustment = -0.05 * max(0, heart_rate - 80)
        
        # Final SpO2 estimate
        spo2 = base_spo2 + hr_adjustment + np.random.normal(0, 0.5)  # Add small noise
        
        return max(70, min(100, spo2))
    
    def update_gui(self):
        """Update GUI with latest results"""
        # Process new results from queue
        while not self.results_buffer.empty():
            try:
                result = self.results_buffer.get_nowait()
                self.add_result(result)
            except queue.Empty:
                break
        
        # Update plots
        self.update_plots()
        
        # Update current readings
        if len(self.spo2_values) > 0:
            current_spo2 = self.spo2_values[-1]
            current_hr = self.heart_rates[-1] if len(self.heart_rates) > 0 else 0
            current_quality = self.quality_scores[-1] if len(self.quality_scores) > 0 else 0
            
            self.spo2_label.config(text=f"{current_spo2:.1f}%")
            self.hr_label.config(text=f"{current_hr:.0f} BPM")
            self.quality_label.config(text=f"{current_quality:.2f}")
            
            # Color coding for SpO2
            if current_spo2 >= 95:
                self.spo2_label.config(foreground="green")
            elif current_spo2 >= 90:
                self.spo2_label.config(foreground="orange")
            else:
                self.spo2_label.config(foreground="red")
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def add_result(self, result):
        """Add new result to storage"""
        self.timestamps.append(result['timestamp'])
        self.spo2_values.append(result['spo2'])
        self.heart_rates.append(result['heart_rate'])
        self.quality_scores.append(result['quality'])
        
        # Keep only last 60 points (1 minute if processing every second)
        max_points = 60
        if len(self.timestamps) > max_points:
            self.timestamps = self.timestamps[-max_points:]
            self.spo2_values = self.spo2_values[-max_points:]
            self.heart_rates = self.heart_rates[-max_points:]
            self.quality_scores = self.quality_scores[-max_points:]
    
    def update_plots(self):
        """Update matplotlib plots"""
        if not hasattr(self, 'canvas'):
            return
        
        if len(self.timestamps) > 0:
            # Update SpO2 plot
            self.spo2_line.set_data(self.timestamps, self.spo2_values)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update HR plot
            self.hr_line.set_data(self.timestamps, self.heart_rates)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Update quality plot
            self.quality_line.set_data(self.timestamps, self.quality_scores)
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            # Redraw canvas
            self.canvas.draw_idle()
    
    def clear_results(self):
        """Clear all results"""
        self.timestamps.clear()
        self.spo2_values.clear()
        self.heart_rates.clear()
        self.quality_scores.clear()
        
        # Clear GUI displays
        self.spo2_label.config(text="--", foreground="black")
        self.hr_label.config(text="--", foreground="black")
        self.quality_label.config(text="--", foreground="black")
        
        # Clear plots
        if hasattr(self, 'canvas'):
            self.spo2_line.set_data([], [])
            self.hr_line.set_data([], [])
            self.quality_line.set_data([], [])
            self.canvas.draw_idle()
    
    def save_results(self):
        """Save processing results to file"""
        if len(self.timestamps) == 0:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                data = {
                    'timestamps': self.timestamps,
                    'spo2_values': self.spo2_values,
                    'heart_rates': self.heart_rates,
                    'quality_scores': self.quality_scores,
                    'summary': {
                        'mean_spo2': np.mean(self.spo2_values),
                        'std_spo2': np.std(self.spo2_values),
                        'mean_hr': np.mean(self.heart_rates),
                        'mean_quality': np.mean(self.quality_scores),
                        'processing_time': datetime.now().isoformat()
                    }
                }
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                elif file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.DataFrame({
                        'timestamp': self.timestamps,
                        'spo2': self.spo2_values,
                        'heart_rate': self.heart_rates,
                        'quality': self.quality_scores
                    })
                    df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Batch processing utility
class PPGBatchProcessor:
    """Utility for batch processing multiple PPG files"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        
    def process_directory(self, input_dir, output_dir, file_pattern="*.csv"):
        """Process all files in a directory"""
        import glob
        import os
        
        input_files = glob.glob(os.path.join(input_dir, file_pattern))
        results = []
        
        print(f"Processing {len(input_files)} files...")
        
        for i, file_path in enumerate(input_files):
            print(f"Processing {i+1}/{len(input_files)}: {os.path.basename(file_path)}")
            
            try:
                # Load data
                if file_path.endswith('.csv'):
                    import pandas as pd
                    data = pd.read_csv(file_path).values
                else:
                    data = np.loadtxt(file_path)
                
                # Process using demo processor (simplified)
                demo = PPGRealtimeDemo()
                if data.ndim == 1:
                    ppg_data = {'ppg_signal': data, 'red_ppg': data, 'ir_ppg': data}
                else:
                    ppg_data = {
                        'ppg_signal': data[:, 0],
                        'red_ppg': data[:, 0],
                        'ir_ppg': data[:, 1] if data.shape[1] > 1 else data[:, 0]
                    }
                
                # Process in segments
                segment_results = []
                window_size = 5 * self.fs
                step_size = 1 * self.fs
                
                for pos in range(0, len(data) - window_size, step_size):
                    window_signal = ppg_data['ppg_signal'][pos:pos + window_size]
                    window_red = ppg_data['red_ppg'][pos:pos + window_size]
                    window_ir = ppg_data['ir_ppg'][pos:pos + window_size]
                    
                    result = demo.process_window(window_signal, window_red, window_ir)
                    result['timestamp'] = pos / self.fs
                    segment_results.append(result)
                
                # Save results
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_results.json")
                
                summary = {
                    'file': file_path,
                    'segments': len(segment_results),
                    'mean_spo2': np.mean([r['spo2'] for r in segment_results]),
                    'mean_hr': np.mean([r['heart_rate'] for r in segment_results]),
                    'mean_quality': np.mean([r['quality'] for r in segment_results])
                }
                
                with open(output_file, 'w') as f:
                    json.dump({
                        'summary': summary,
                        'results': segment_results
                    }, f, indent=2)
                
                results.append(summary)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Save batch summary
        batch_summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(batch_summary_file, 'w') as f:
            json.dump({
                'processed_files': len(results),
                'processing_time': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"Batch processing completed. Results saved to {output_dir}")
        return results

# Main execution
def main():
    """Main function to run the demo"""
    print("PPG Processing Demo Application")
    print("=" * 40)
    print("Choose mode:")
    print("1. Real-time GUI Demo")
    print("2. Batch Processing")
    print("3. Generate sample data and exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("Starting GUI demo...")
        app = PPGRealtimeDemo()
        app.run()
    
    elif choice == "2":
        input_dir = input("Enter input directory path: ").strip()
        output_dir = input("Enter output directory path: ").strip()
        
        if not input_dir or not output_dir:
            print("Invalid directories specified")
            return
        
        processor = PPGBatchProcessor()
        results = processor.process_directory(input_dir, output_dir)
        print(f"Processed {len(results)} files successfully")
    
    elif choice == "3":
        print("Generating sample PPG data...")
        
        # Generate sample data
        demo = PPGRealtimeDemo()
        demo.generate_demo_data()
        
        # Save to files
        np.save("sample_red_ppg.npy", demo.ppg_data['red_ppg'])
        np.save("sample_ir_ppg.npy", demo.ppg_data['ir_ppg'])
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame({
            'red_ppg': demo.ppg_data['red_ppg'],
            'ir_ppg': demo.ppg_data['ir_ppg']
        })
        df.to_csv("sample_ppg_data.csv", index=False)
        
        print("Sample data saved:")
        print("- sample_red_ppg.npy")
        print("- sample_ir_ppg.npy") 
        print("- sample_ppg_data.csv")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()