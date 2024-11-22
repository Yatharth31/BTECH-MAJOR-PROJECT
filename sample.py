from TTS.api import TTS

def text_to_audio(text, output_file="output_audio.wav", speaker=None, language=None):
    """
    Converts input text to speech using Coqui TTS.

    Parameters:
        text (str): The text to be converted into speech.
        output_file (str): The output file path for the generated audio (default: "output_audio.wav").
        speaker (str): The speaker ID for multi-speaker models (default: None).
        language (str): The language ID for multilingual models (default: None).

    Returns:
        None: The audio file is saved to the specified path.
    """
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)

    # Generate speech and save the audio file
    tts.tts_to_file(text=text, speaker=speaker, language=language, file_path=output_file)
    print(f"Audio saved as {output_file}")


if __name__ == "__main__":
    user_text = "Hello! This is a sample text being converted to speech using Coqui TTS."
    output_audio_file = "sample_audio.wav"

    text_to_audio(user_text, output_file=output_audio_file)
