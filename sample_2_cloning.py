from TTS.api import TTS

def text_to_speech_with_cloning(text, output_file="cloned_voice_audio.wav", speaker_wav_path=None,language="en"):
    """
    Converts text to speech using voice cloning with Coqui TTS.

    Parameters:
        text (str): The text to be converted into speech.
        output_file (str): The output file path for the generated audio.
        speaker_wav_path (str): Path to a .wav file for cloning the speaker's voice (required for voice cloning).

    Returns:
        None: The audio file is saved to the specified path.
    """
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)

    if not speaker_wav_path:
        raise ValueError("You must provide a path to a .wav file for voice cloning.")
    
    tts.tts_to_file(text=text, speaker_wav=speaker_wav_path,language=language, file_path=output_file)
    print(f"Audio with cloned voice saved as {output_file}")


if __name__ == "__main__":
    output_audio_file = "E:/1_BTech/Major_Project/FINAL/cloned_voice.wav"

    text_to_speech_with_cloning("Clubs and Balls and cities, Grew to be only memories", output_file=output_audio_file, speaker_wav_path=f"E:/1_BTech/Major_Project/FINAL/train_marathifemale_00955.wav",language="en")