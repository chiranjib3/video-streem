import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, ifft
import librosa
import soundfile as sf
import threading
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Advanced real-time audio processing for voice enhancement"""
    
    def __init__(self, sample_rate=48000, frame_size=1024):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_profile = None
        self.audio_buffer = deque(maxlen=10)  # Keep last 10 frames for processing
        
        # Processing parameters
        self.noise_reduction_strength = 0.7
        self.voice_enhancement_enabled = True
        self.auto_gain_enabled = True
        self.echo_cancellation_enabled = True
        
        # Filters and processors
        self.setup_filters()
        
    def setup_filters(self):
        """Initialize audio filters and processors"""
        # High-pass filter to remove low-frequency noise
        self.highpass_b, self.highpass_a = signal.butter(
            4, 80, btype='high', fs=self.sample_rate
        )
        
        # Low-pass filter to remove high-frequency noise
        self.lowpass_b, self.lowpass_a = signal.butter(
            4, 8000, btype='low', fs=self.sample_rate
        )
        
        # Notch filter for 50/60 Hz hum removal
        self.notch_60_b, self.notch_60_a = signal.iirnotch(60, 30, self.sample_rate)
        self.notch_50_b, self.notch_50_a = signal.iirnotch(50, 30, self.sample_rate)
        
        # Voice frequency emphasis filter (300-3400 Hz)
        self.voice_b, self.voice_a = signal.butter(
            4, [300, 3400], btype='band', fs=self.sample_rate
        )
    
    def spectral_subtraction(self, audio_data):
        """Apply spectral subtraction for noise reduction"""
        try:
            # Convert to frequency domain
            fft_data = fft(audio_data)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # Estimate noise if not available
            if self.noise_profile is None:
                self.noise_profile = magnitude * 0.1  # Conservative estimate
            
            # Calculate spectral subtraction
            alpha = self.noise_reduction_strength
            beta = 0.1  # Over-subtraction factor
            
            # Subtract noise estimate
            clean_magnitude = magnitude - alpha * self.noise_profile
            
            # Apply spectral floor to prevent artifacts
            spectral_floor = beta * magnitude
            clean_magnitude = np.maximum(clean_magnitude, spectral_floor)
            
            # Reconstruct signal
            clean_fft = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.real(ifft(clean_fft))
            
            return clean_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Spectral subtraction error: {e}")
            return audio_data
    
    def adaptive_noise_reduction(self, audio_data):
        """Adaptive noise reduction using statistical methods"""
        try:
            # Calculate moving average for noise estimation
            if len(self.audio_buffer) > 5:
                # Estimate noise from quiet periods
                energy_threshold = np.mean([np.mean(frame**2) for frame in self.audio_buffer]) * 0.1
                current_energy = np.mean(audio_data**2)
                
                if current_energy < energy_threshold:
                    # Update noise profile during quiet periods
                    fft_data = fft(audio_data)
                    current_noise = np.abs(fft_data)
                    
                    if self.noise_profile is None:
                        self.noise_profile = current_noise
                    else:
                        # Exponential moving average
                        self.noise_profile = 0.9 * self.noise_profile + 0.1 * current_noise
            
            # Apply spectral subtraction
            return self.spectral_subtraction(audio_data)
            
        except Exception as e:
            logger.error(f"Adaptive noise reduction error: {e}")
            return audio_data
    
    def voice_enhancement(self, audio_data):
        """Enhance voice frequencies and reduce non-voice content"""
        try:
            # Apply voice frequency emphasis
            enhanced = signal.filtfilt(self.voice_b, self.voice_a, audio_data)
            
            # Dynamic range compression for voice
            # Soft knee compression
            threshold = 0.1
            ratio = 3.0
            
            # Apply compression
            compressed = np.copy(enhanced)
            over_threshold = np.abs(enhanced) > threshold
            
            if np.any(over_threshold):
                # Compress signals above threshold
                sign = np.sign(enhanced[over_threshold])
                abs_vals = np.abs(enhanced[over_threshold])
                compressed_vals = threshold + (abs_vals - threshold) / ratio
                compressed[over_threshold] = sign * compressed_vals
            
            return compressed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Voice enhancement error: {e}")
            return audio_data
    
    def auto_gain_control(self, audio_data):
        """Automatic gain control to normalize audio levels"""
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Target RMS level
            target_rms = 0.1
            
            if rms > 0:
                # Calculate gain
                gain = target_rms / rms
                
                # Limit gain to prevent excessive amplification
                gain = np.clip(gain, 0.1, 5.0)
                
                # Apply gain with smooth transitions
                return audio_data * gain
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Auto gain control error: {e}")
            return audio_data
    
    def remove_dc_offset(self, audio_data):
        """Remove DC offset from audio signal"""
        try:
            return audio_data - np.mean(audio_data)
        except Exception as e:
            logger.error(f"DC offset removal error: {e}")
            return audio_data
    
    def apply_windowing(self, audio_data):
        """Apply windowing to reduce spectral leakage"""
        try:
            window = signal.windows.hann(len(audio_data))
            return audio_data * window
        except Exception as e:
            logger.error(f"Windowing error: {e}")
            return audio_data
    
    def process_audio_frame(self, audio_data):
        """Process a single frame of audio data"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, (list, tuple)):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure correct data type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Store frame for adaptive processing
            self.audio_buffer.append(audio_data.copy())
            
            # Processing pipeline
            processed = audio_data.copy()
            
            # 1. Remove DC offset
            processed = self.remove_dc_offset(processed)
            
            # 2. Apply basic filtering
            # Remove low-frequency noise
            processed = signal.filtfilt(self.highpass_b, self.highpass_a, processed)
            
            # Remove high-frequency noise
            processed = signal.filtfilt(self.lowpass_b, self.lowpass_a, processed)
            
            # Remove electrical hum
            processed = signal.filtfilt(self.notch_60_b, self.notch_60_a, processed)
            processed = signal.filtfilt(self.notch_50_b, self.notch_50_a, processed)
            
            # 3. Noise reduction
            processed = self.adaptive_noise_reduction(processed)
            
            # 4. Voice enhancement
            if self.voice_enhancement_enabled:
                processed = self.voice_enhancement(processed)
            
            # 5. Auto gain control
            if self.auto_gain_enabled:
                processed = self.auto_gain_control(processed)
            
            # 6. Final limiting to prevent clipping
            processed = np.clip(processed, -1.0, 1.0)
            
            return processed
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_data
    
    def reset_noise_profile(self):
        """Reset the noise profile for re-calibration"""
        self.noise_profile = None
        self.audio_buffer.clear()
        logger.info("Noise profile reset")
    
    def update_settings(self, settings):
        """Update processing settings"""
        if 'noise_reduction_strength' in settings:
            self.noise_reduction_strength = np.clip(settings['noise_reduction_strength'], 0.0, 1.0)
        
        if 'voice_enhancement_enabled' in settings:
            self.voice_enhancement_enabled = settings['voice_enhancement_enabled']
        
        if 'auto_gain_enabled' in settings:
            self.auto_gain_enabled = settings['auto_gain_enabled']
        
        if 'echo_cancellation_enabled' in settings:
            self.echo_cancellation_enabled = settings['echo_cancellation_enabled']
        
        logger.info(f"Audio processor settings updated: {settings}")


class RealTimeAudioProcessor:
    """Real-time audio processor for live streaming"""
    
    def __init__(self, sample_rate=48000, frame_size=1024):
        self.processor = AudioProcessor(sample_rate, frame_size)
        self.is_running = False
        self.input_queue = deque(maxlen=100)
        self.output_queue = deque(maxlen=100)
        self.processing_thread = None
        
    def start(self):
        """Start real-time processing"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Real-time audio processor started")
    
    def stop(self):
        """Stop real-time processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Real-time audio processor stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                if self.input_queue:
                    audio_frame = self.input_queue.popleft()
                    processed_frame = self.processor.process_audio_frame(audio_frame)
                    self.output_queue.append(processed_frame)
                else:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def add_input_frame(self, audio_data):
        """Add audio frame for processing"""
        try:
            self.input_queue.append(audio_data)
        except Exception as e:
            logger.error(f"Error adding input frame: {e}")
    
    def get_output_frame(self):
        """Get processed audio frame"""
        try:
            if self.output_queue:
                return self.output_queue.popleft()
            return None
        except Exception as e:
            logger.error(f"Error getting output frame: {e}")
            return None
    
    def update_settings(self, settings):
        """Update processor settings"""
        self.processor.update_settings(settings)
    
    def reset_noise_profile(self):
        """Reset noise profile"""
        self.processor.reset_noise_profile()


# Audio quality metrics
def calculate_snr(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio"""
    try:
        noise = noisy_signal - clean_signal
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        return float('inf')
    except Exception as e:
        logger.error(f"SNR calculation error: {e}")
        return 0


def analyze_audio_quality(audio_data, sample_rate=48000):
    """Analyze audio quality metrics"""
    try:
        metrics = {}
        
        # RMS level
        metrics['rms_level'] = np.sqrt(np.mean(audio_data**2))
        
        # Peak level
        metrics['peak_level'] = np.max(np.abs(audio_data))
        
        # Dynamic range
        metrics['dynamic_range'] = metrics['peak_level'] / (metrics['rms_level'] + 1e-10)
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        metrics['spectral_centroid'] = np.mean(spectral_centroids)
        
        # Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        metrics['zero_crossing_rate'] = np.mean(zcr)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return {}
