from flask import Flask, request, jsonify, render_template
import os
import threading
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from safetensors import safe_open
from speech_to_text import speech_to_text, infer_language, speech_activity_detection, transcribe_audio
from text_to_speech import text_to_speech_with_cloning
from text_to_text import text_to_text_translation
from Text_to_ISL.convert2isl import convert_to_isl
from audio_enhancement import convert_to_wav

app = Flask(__name__)
# Initialize model and processor variables globally
model = None
processor = None

def load_model():
    global model, processor
    try:
        print("Loading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        safetensor_path = "E:/1_BTech/Major_Project/FINAL/model/model_whisper.safetensors"
        tensors = {}
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")




@app.route("/")
def index():
    return render_template("index.html")  


@app.route("/process", methods=["POST"])
def process_audio():
    if model is None or processor is None:
        return jsonify({"error": "Model is not loaded correctly"}), 500
    if "audio_file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    if not file_path.lower().endswith('.wav'):
        print("Converting File to wav format...")
        try:
            file_path = convert_to_wav(file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # Process audio file
    try:
        if speech_activity_detection(file_path) is None:
            print("No speech")
            return jsonify({"error": "No speech in audio file"}), 400
        result = speech_to_text(file_path)
        speech_text = result["text"]
        print("Transcription:", speech_text)

        # Translation
        input_language = infer_language(speech_text)
        output_language = "en"
        translated_text = text_to_text_translation(speech_text, input_language, output_language)
        print("Translated Text:", translated_text)

        # Start TTS and ISL in separate threads
        def run_tts(translated_text, file_path):
            text_to_speech_with_cloning(
                translated_text,
                output_file="cloned_voice.wav", 
                speaker_wav_path=file_path,
                language="en",
            )

        def run_text_to_isl(translated_text):
            convert_to_isl(translated_text)

        tts_thread = threading.Thread(target=run_tts, args=(translated_text, file_path))
        isl_thread = threading.Thread(target=run_text_to_isl, args=(translated_text,))

        tts_thread.start()
        isl_thread.start()

        tts_thread.join()
        isl_thread.join()

        return jsonify({
            "transcription": speech_text,
            "translation": translated_text,
            "tts_audio": "cloned_voice.wav",
            "isl_video": "ISL video generated successfully."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
