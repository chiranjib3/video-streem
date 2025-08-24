from flask import Blueprint, request, jsonify, session
from flask_socketio import emit
import numpy as np
import base64
import io
import wave
import json
import logging
from audio_processor import AudioProcessor, RealTimeAudioProcessor, analyze_audio_quality

logger = logging.getLogger(__name__)

# Create blueprint for audio API
audio_bp = Blueprint('audio', __name__)

# Global audio processors for each session
audio_processors = {}

def get_or_create_processor(session_id):
    """Get or create audio processor for session"""
    if session_id not in audio_processors:
        audio_processors[session_id] = RealTimeAudioProcessor()
        audio_processors[session_id].start()
    return audio_processors[session_id]

def cleanup_processor(session_id):
    """Cleanup audio processor for session"""
    if session_id in audio_processors:
        audio_processors[session_id].stop()
        del audio_processors[session_id]

@audio_bp.route('/api/audio/enhance', methods=['POST'])
def enhance_audio():
    """Enhance audio data endpoint"""
    try:
        data = request.get_json()
        
        if 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(data['audio_data'])
        
        # Convert to numpy array (assuming 16-bit PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Get or create processor
        session_id = session.get('session_id', request.remote_addr)
        processor = get_or_create_processor(session_id)
        
        # Process audio
        enhanced_audio = processor.processor.process_audio_frame(audio_array)
        
        # Convert back to bytes
        enhanced_bytes = (enhanced_audio * 32768.0).astype(np.int16).tobytes()
        enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
        
        # Calculate quality metrics
        quality_metrics = analyze_audio_quality(enhanced_audio)
        
        return jsonify({
            'enhanced_audio': enhanced_b64,
            'quality_metrics': quality_metrics,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Audio enhancement error: {e}")
        return jsonify({'error': str(e)}), 500

@audio_bp.route('/api/audio/settings', methods=['POST'])
def update_audio_settings():
    """Update audio processing settings"""
    try:
        data = request.get_json()
        session_id = session.get('session_id', request.remote_addr)
        
        if session_id in audio_processors:
            audio_processors[session_id].update_settings(data)
            return jsonify({'success': True, 'message': 'Settings updated'})
        else:
            return jsonify({'error': 'No active audio processor'}), 400
            
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return jsonify({'error': str(e)}), 500

@audio_bp.route('/api/audio/reset_noise_profile', methods=['POST'])
def reset_noise_profile():
    """Reset noise profile for recalibration"""
    try:
        session_id = session.get('session_id', request.remote_addr)
        
        if session_id in audio_processors:
            audio_processors[session_id].reset_noise_profile()
            return jsonify({'success': True, 'message': 'Noise profile reset'})
        else:
            return jsonify({'error': 'No active audio processor'}), 400
            
    except Exception as e:
        logger.error(f"Noise profile reset error: {e}")
        return jsonify({'error': str(e)}), 500

@audio_bp.route('/api/audio/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio quality"""
    try:
        data = request.get_json()
        
        if 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(data['audio_data'])
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Analyze quality
        quality_metrics = analyze_audio_quality(audio_array)
        
        return jsonify({
            'quality_metrics': quality_metrics,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return jsonify({'error': str(e)}), 500

# SocketIO handlers for real-time audio processing
def register_audio_socketio_handlers(socketio):
    """Register SocketIO handlers for audio processing"""
    
    @socketio.on('audio_frame')
    def handle_audio_frame(data):
        """Handle real-time audio frame processing"""
        try:
            session_id = session.get('session_id', request.sid)
            
            # Get or create processor
            processor = get_or_create_processor(session_id)
            
            # Decode audio data
            if 'audio_data' in data:
                audio_bytes = base64.b64decode(data['audio_data'])
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to processing queue
                processor.add_input_frame(audio_array)
                
                # Get processed frame if available
                processed_frame = processor.get_output_frame()
                if processed_frame is not None:
                    # Convert back to bytes and base64
                    processed_bytes = (processed_frame * 32768.0).astype(np.int16).tobytes()
                    processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
                    
                    emit('processed_audio_frame', {
                        'audio_data': processed_b64,
                        'timestamp': data.get('timestamp', 0)
                    })
            
        except Exception as e:
            logger.error(f"Audio frame processing error: {e}")
            emit('audio_error', {'error': str(e)})
    
    @socketio.on('start_audio_processing')
    def handle_start_audio_processing(data):
        """Start audio processing for session"""
        try:
            session_id = session.get('session_id', request.sid)
            processor = get_or_create_processor(session_id)
            
            # Update settings if provided
            if 'settings' in data:
                processor.update_settings(data['settings'])
            
            emit('audio_processing_started', {'success': True})
            logger.info(f"Audio processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"Start audio processing error: {e}")
            emit('audio_error', {'error': str(e)})
    
    @socketio.on('stop_audio_processing')
    def handle_stop_audio_processing():
        """Stop audio processing for session"""
        try:
            session_id = session.get('session_id', request.sid)
            cleanup_processor(session_id)
            
            emit('audio_processing_stopped', {'success': True})
            logger.info(f"Audio processing stopped for session {session_id}")
            
        except Exception as e:
            logger.error(f"Stop audio processing error: {e}")
            emit('audio_error', {'error': str(e)})
    
    @socketio.on('update_audio_settings')
    def handle_update_audio_settings(data):
        """Update audio processing settings via SocketIO"""
        try:
            session_id = session.get('session_id', request.sid)
            
            if session_id in audio_processors:
                audio_processors[session_id].update_settings(data.get('settings', {}))
                emit('audio_settings_updated', {'success': True})
            else:
                emit('audio_error', {'error': 'No active audio processor'})
                
        except Exception as e:
            logger.error(f"Update audio settings error: {e}")
            emit('audio_error', {'error': str(e)})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Cleanup on disconnect"""
        try:
            session_id = session.get('session_id', request.sid)
            cleanup_processor(session_id)
        except Exception as e:
            logger.error(f"Disconnect cleanup error: {e}")


# Audio processing utilities
class AudioBuffer:
    """Circular buffer for audio data"""
    
    def __init__(self, max_size=10):
        self.buffer = []
        self.max_size = max_size
        self.index = 0
    
    def add(self, data):
        """Add data to buffer"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
            self.index = (self.index + 1) % self.max_size
    
    def get_all(self):
        """Get all data in buffer"""
        if len(self.buffer) < self.max_size:
            return self.buffer
        else:
            return self.buffer[self.index:] + self.buffer[:self.index]
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.index = 0


def process_wav_file(file_path, output_path=None):
    """Process WAV file with audio enhancement"""
    try:
        # Read WAV file
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
        
        # Convert to numpy array
        if sample_width == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)
        elif sample_width == 4:
            audio_data = np.frombuffer(frames, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Handle stereo
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        
        # Normalize to float32
        audio_data = audio_data.astype(np.float32)
        if sample_width == 2:
            audio_data /= 32768.0
        elif sample_width == 4:
            audio_data /= 2147483648.0
        
        # Process audio
        processor = AudioProcessor(sample_rate=sample_rate)
        
        # Process in chunks to handle long files
        chunk_size = 1024
        processed_chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            processed_chunk = processor.process_audio_frame(chunk)
            processed_chunks.append(processed_chunk)
        
        # Combine processed chunks
        processed_audio = np.concatenate(processed_chunks)
        
        # Convert back to original format
        if sample_width == 2:
            processed_audio = (processed_audio * 32768.0).astype(np.int16)
        elif sample_width == 4:
            processed_audio = (processed_audio * 2147483648.0).astype(np.int32)
        
        # Save processed audio
        if output_path:
            with wave.open(output_path, 'wb') as output_wav:
                output_wav.setnchannels(1)  # Mono output
                output_wav.setsampwidth(sample_width)
                output_wav.setframerate(sample_rate)
                output_wav.writeframes(processed_audio.tobytes())
        
        return processed_audio
        
    except Exception as e:
        logger.error(f"WAV file processing error: {e}")
        raise


def create_audio_test_signals():
    """Create test audio signals for demonstration"""
    sample_rate = 48000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean speech-like signal
    clean_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 tone
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 tone
        0.2 * np.sin(2 * np.pi * 1320 * t)   # E6 tone
    ) * np.exp(-t / 2)  # Decay envelope
    
    # Add various types of noise
    noise = (
        0.1 * np.random.normal(0, 1, len(t)) +  # White noise
        0.05 * np.sin(2 * np.pi * 60 * t) +    # 60Hz hum
        0.03 * np.sin(2 * np.pi * 120 * t) +   # 120Hz harmonic
        0.02 * np.random.normal(0, 1, len(t)) * np.sin(2 * np.pi * 10 * t)  # Modulated noise
    )
    
    noisy_signal = clean_signal + noise
    
    return clean_signal, noisy_signal, sample_rate
