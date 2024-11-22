import requests
from langdetect import detect
from Text_to_Text.text_to_text import text_to_text_translation
from Text_to_Speech.TTS.sample_2_cloning import text_to_speech_with_cloning
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer hf_OvPKZFkpRMkistcJJvArQmNvjCIZSAuBfzj"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

speaker_audio_path = "./train_marathifemale_00955.wav"
output = query(speaker_audio_path)
# print(output)


def infer_language(transcription):
    return detect(transcription)

language = infer_language(output["text"])

print("Detected Language:", language)


# Extracting information
if "text" in output and language:
    print("Detected Language:", language)
    print("Transcription Text:", output["text"])
else:
    print("Response:", output)


transalted_text = text_to_text_translation(output["text"],language,output_language="en")
print(transalted_text)

output_audio_file = "E:/1_BTech/Major_Project/Developement-Code/cloned_voice_sample.wav"

text_to_speech_with_cloning(transalted_text, output_file=output_audio_file, speaker_wav_path=speaker_audio_path,language="en")