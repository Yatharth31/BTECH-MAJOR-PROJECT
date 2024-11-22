import torch
import os
from silero_vad import get_speech_timestamps, read_audio
import warnings
import librosa
from langdetect import detect
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("LOCATION")
token = os.getenv("HF_TOKEN")
# Global model variable to hold the model
model = None
utils = None

# Load model only once when the Flask app starts
def load_model():
    global model, utils
    print("Loading model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad', 
        force_reload=False  # Cache the model
    )
    print("Model loaded successfully!")

# Call load_model function when the app starts
load_model()

# Extract utils for later use
get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils


# Function to transcribe audio using Whisper model
def transcribe_audio(file_path, model, processor):
    warnings.filterwarnings('ignore')
    print("Transcribing...")
    audio_array, sampling_rate = librosa.load(file_path, sr=16000)
    
    input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

    with torch.no_grad():
        generated_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Language detection from transcription
def infer_language(transcription):
    return detect(transcription)

# API call for additional speech processing (external API)
def speech_to_text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(
        url, 
        headers={"Authorization": f"Bearer {token}"},  # Ensure using a token from .env file
        data=data
    )
    print(response.json())
    return response.json()

# Speech Activity Detection using Silero VAD
def speech_activity_detection(filepath):
    wav = read_audio(filepath, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model)
    
    if speech_timestamps:
        return True
    else:
        print("No speech detected.")
        return None
