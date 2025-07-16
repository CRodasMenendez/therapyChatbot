import sounddevice as sd
from scipy.io.wavfile import write
import time
import numpy as np
import whisper  

#Load Whisper model for transcription 
model = whisper.load_model("base")

#Function to record audio and save it to a file
def record_audio():
    print("Please Press Enter to Start Recording...")
    input() #waits for enter to be pressed to start

    fs = 44100 #sample rate (standard)
    channels = 1 #number of channels 

    # Record for a maximum of 300 seconds (5 minutes)
    max_duration = 300
    recording = sd.rec(int(fs * max_duration), samplerate=fs, channels=channels)

    print("recording, press enter to stop \n")
    
    start_time = time.time()
    input() #waits for enter to be pressed again to stop recording
    end_time = time.time()

    print("stopping... ")

    sd.stop()

    #calculate duration
    duration = end_time - start_time
    samples_recorded = int(duration * fs)
    
    #save recorded portion

    # Extract only the portion that was actually recorded
    if samples_recorded > 0 and samples_recorded <= len(recording):
        actual_recording = recording[:samples_recorded]

        write("recording.wav", fs, actual_recording) 
        #print(f"Audio saved: {duration:.2f} seconds")
        return True
    else:
        print("No audio recorded")
        return False

#function to translate audio file to text 
def speech_to_text():
    try:
        result = model.transcribe("recording.wav", language="en") 
        return result['text']
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

#Function to transcribe audio
def transcribe():
    success = record_audio()
        
    if success:
        #print("Transcribing... \n")
        output = speech_to_text()

        # Check for actual text content
        if output and output.strip():
            #print("\n Transcription: ")
            #print(output)
            return output
        else:
            print("Transcription Failed - No speech detected")


if __name__ == "__main__":
    transcribe()