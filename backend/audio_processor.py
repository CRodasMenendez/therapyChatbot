# backend/audio_processor.py
import whisper
import io
import wave
import numpy as np
from typing import Optional
import tempfile
import os

class AudioProcessor:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        print("whisper model loaded successfully")
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        try:
            #create temporary file to store audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                #transcribe the audio file
                result = self.model.transcribe(temp_file_path, language="en")
                transcribed_text = result['text'].strip()
                
                #clean up the temporary file
                os.unlink(temp_file_path)
                
                #return transcription if it contains actual content
                if transcribed_text and len(transcribed_text) > 1:
                    return transcribed_text
                else:
                    return None
                    
            except Exception as e:
                #clean up temp file if transcription fails
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise e
                
        except Exception as e:
            print(f"error during transcription: {e}")
            return None
    
    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        try:
            #create temporary file with wave format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            #write audio array to WAV file
            with wave.open(temp_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                #convert float array to 16-bit integers if necessary
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)
                
                wav_file.writeframes(audio_array.tobytes())
            
            try:
                #transcribe the audio file
                result = self.model.transcribe(temp_file_path, language="en")
                transcribed_text = result['text'].strip()
                
                #clean up
                os.unlink(temp_file_path)
                
                return transcribed_text if transcribed_text else None
                
            except Exception as e:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise e
                
        except Exception as e:
            print(f"Error processing audio array: {e}")
            return None
    
    def validate_audio_format(self, audio_data: bytes) -> bool:
        try:
            #check if it's a WAV file by header
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                return True
            
            #basic size check for other formats
            return len(audio_data) > 1000
            
        except Exception:
            return False
    
    def get_audio_duration(self, audio_data: bytes) -> Optional[float]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                with wave.open(temp_file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / float(sample_rate)
                
                os.unlink(temp_file_path)
                return duration
                
            except Exception:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return None
                
        except Exception:
            return None