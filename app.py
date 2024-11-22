import torch
import librosa
import warnings 
import requests
import threading
from langdetect import detect
from speech_to_text import transcribe_audio, speech_to_text, infer_language
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from safetensors import safe_open
from audio_enhancement import enhance_audio
from text_to_speech import text_to_speech_with_cloning
from text_to_text import text_to_text_translation
from Text_to_ISL.convert2isl import convert_to_isl

warnings.filterwarnings('ignore') 


if __name__=="__main__":
    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")  # Use any model architecture

    safetensor_path = "E:/1_BTech/Major_Project/Developement-Code/FINAL/model/model.safetensors"  # Path to your saved safetensors file
    tensors = {}

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print("Model loaded successfully...")
    
    audio_file_path = "./train_hindimale_00650.wav" 
    result = speech_to_text(audio_file_path)
    speech_text = result["text"]
    print("Transcription:", speech_text)
    
    print("Translation to desired language...")
    input_language = infer_language(speech_text)
    print("Detected Language:", input_language)
    output_language = "en"
    translated_text = text_to_text_translation(speech_text, input_language, output_language)
    print("Translated Text:", translated_text)
    
    def run_tts(translated_text, audio_file_path):
        print("TTS in progress...")
        text_to_speech_with_cloning(
            translated_text,
            output_file="cloned_voice.wav",
            speaker_wav_path=audio_file_path,
            language="en",
        )
        print("TTS completed successfully.")

    def run_text_to_isl(translated_text):
        print("Text to ISL in progress...")
        convert_to_isl(translated_text)
        print("ISL video generated successfully.")

    tts_thread = threading.Thread(target=run_tts, args=(translated_text, audio_file_path))
    isl_thread = threading.Thread(target=run_text_to_isl, args=(translated_text,))

    tts_thread.start()
    isl_thread.start()

    tts_thread.join()
    isl_thread.join()

    print("Both TTS and Text to ISL processes are complete.")