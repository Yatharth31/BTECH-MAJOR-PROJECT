import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from safetensors import safe_open
import librosa
from langdetect import detect
import warnings 
import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("LOCATION")
token = os.getenv("HF_TOKEN")
    
def transcribe_audio(file_path,model,processor):
    # Settings the warnings to be ignored 
    warnings.filterwarnings('ignore') 

    print("Transcribing...")
    audio_array, sampling_rate = librosa.load(file_path, sr=16000)
    
    input_features = processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    with torch.no_grad():
        generated_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("\n\n\n\nTranscription:", transcription, "\n\n\n\n")
    return transcription

def infer_language(transcription):
    return detect(transcription)

def speech_to_text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(url, headers={"Authorization": "Bearer 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"}, data=data)
    return response.json()