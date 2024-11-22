import torch
import numpy as np
import wave
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from torchmetrics.audio import SignalDistortionRatio
import librosa
import soundfile as sf
from denoiser import pretrained, enhance
from denoiser.stft_loss import MultiResolutionSTFTLoss
import threading

# Function to convert other format to .wav
def convert_to_wav(input_path):
    try:
        if not input_path.lower().endswith('.wav'):
            audio_data, sample_rate = librosa.load(input_path, sr=None)
            output_path = os.path.splitext(input_path)[0] + '.wav'
            sf.write(output_path, audio_data, sample_rate)

            print(f"Converted {input_path} to {output_path}")
            return output_path
        
        return input_path

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None

# Function to read a .wav file using librosa
def read_wave_file(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    return audio_data, sample_rate

# Function to save a .wav file
def save_wave_file(output_path, audio_data, sample_rate):
    audio_data = (audio_data * 32768.0).astype(np.int16)  # Convert from float32 to int16 (16-bit PCM)
    
    # Open the file for writing
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # (1 for mono audio)
        wav_file.setsampwidth(2) 
        wav_file.setframerate(sample_rate) 
        wav_file.writeframes(audio_data.tobytes())

    print(f"Saved denoised audio to {output_path}")

# Function to convert multi-channel audio to mono by averaging
def convert_to_mono(denoised_waveform):
    if len(denoised_waveform.shape) == 2:
        # Average across channels to convert to mono
        denoised_waveform = denoised_waveform.mean(axis=0)
    return denoised_waveform

# Function to denoise the audio using Facebook's Denoiser
def denoise_audio(model, noisy_waveform, sample_rate):
    noisy_waveform_tensor = torch.from_numpy(noisy_waveform).unsqueeze(0)
    with torch.no_grad():
        denoised_waveform_tensor = model(noisy_waveform_tensor)
    denoised_waveform_np = denoised_waveform_tensor.squeeze(0).numpy()
    print(denoised_waveform_np.shape)  # Ensure it's a 1D array for mono or 2D for stereo
    # Convert to mono if the waveform is multi-channel
    denoised_waveform_np = convert_to_mono(denoised_waveform_np)
    return denoised_waveform_np

# Function to plot and save waveforms as JPG
def plot_waveforms(original_waveform, denoised_waveform, sample_rate, output_filename="waveform_plot.jpg"):
    plt.figure(figsize=(18, 8))

    # Plot original waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(original_waveform) / sample_rate, num=len(original_waveform)), original_waveform)
    plt.title('Original Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    # Plot denoised waveform
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(denoised_waveform) / sample_rate, num=len(denoised_waveform)), denoised_waveform)
    plt.title('Denoised Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    # Save plot as JPG
    plt.savefig(output_filename, format='jpg')


# Function to calculate SDR loss
def calculate_sdr_loss(original_waveform, denoised_waveform):
    # Remove any extra dimensions from the denoised waveform
    denoised_waveform = denoised_waveform.squeeze()

    # Initialize the SDR metric
    sdr = SignalDistortionRatio()

    # Convert the waveforms to tensors
    original_tensor = torch.tensor(original_waveform)
    denoised_tensor = torch.tensor(denoised_waveform)

    # Calculate SDR loss
    sdr_loss = sdr(original_tensor, denoised_tensor).item()

    print(f"SDR Loss: {sdr_loss:.4f}")
    return sdr_loss

def calculate_stft(waveform, sample_rate, fft_size=1024, hop_size=512, win_length=1024, window="hann"):
    """Calculate and return the Short-Time Fourier Transform (STFT) of the waveform.
    Args:
        waveform (ndarray or Tensor): Input audio waveform.
        sample_rate (int): Sampling rate of the waveform.
        fft_size (int): FFT size.
        hop_size (int): Hop size between consecutive frames.
        win_length (int): Window length.
        window (str): Type of window function, e.g., 'hann'.
    Returns:
        stft_result (Tensor): Magnitude of the STFT.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.tensor(waveform)

    # Create the window function
    window_function = torch.hann_window(win_length) if window == "hann" else torch.ones(win_length)

    # Calculate the STFT
    stft_result = torch.stft(
        waveform, 
        n_fft=fft_size, 
        hop_length=hop_size, 
        win_length=win_length, 
        window=window_function, 
        return_complex=True 
    )

    # Compute the magnitude of the STFT (using the absolute value of the complex STFT)
    magnitude = torch.abs(stft_result)
    
    return magnitude

def stft_thread_function(waveform, sample_rate):
    stft_result = calculate_stft(waveform, sample_rate)
    print("STFT calculation completed.") 
    return stft_result

def calculate_stft_loss(predicted_waveform, target_waveform, sample_rate):
    try:
        if isinstance(predicted_waveform, np.ndarray):
            predicted_waveform = torch.tensor(predicted_waveform)
        if isinstance(target_waveform, np.ndarray):
            target_waveform = torch.tensor(target_waveform)
            
        predicted_waveform = predicted_waveform.unsqueeze(0) if len(predicted_waveform.shape) == 1 else predicted_waveform
        target_waveform = target_waveform.unsqueeze(0) if len(target_waveform.shape) == 1 else target_waveform
        # Initialize the multi-resolution STFT loss
        stft_loss_module = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512], 
            hop_sizes=[120, 240, 50], 
            win_lengths=[600, 1200, 240], 
            window="hann_window"
        )
        print("STFT loss module initialized.")
        sc_loss, mag_loss = stft_loss_module(predicted_waveform, target_waveform)
        return sc_loss, mag_loss
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# Function to run STFT loss calculation in a separate thread
def stft_loss_thread_function(predicted_waveform, target_waveform, sample_rate):
    sc_loss, mag_loss = calculate_stft_loss(predicted_waveform, target_waveform, sample_rate)
    print(f"Spectral Convergence Loss: {sc_loss.item():.4f}, Magnitude Loss: {mag_loss.item():.4f}")
    return sc_loss, mag_loss

# Modified enhance_audio function with threading
def enhance_audio(input_path, original_audio_path,model, output_path="denoised_audio.wav"):
    # Convert to WAV if necessary
    wav_path = convert_to_wav(input_path)
    # Load 
    noisy_waveform, sample_rate = read_wave_file(wav_path)
    # Clean audio
    clean_waveform, sample_rate = read_wave_file(original_audio_path)
    # Denoise
    denoised_waveform = denoise_audio(model, noisy_waveform, sample_rate)
    # Save
    save_wave_file(output_path, denoised_waveform, sample_rate)

    threads = []
    # plot_thread = threading.Thread(target=plot_waveforms, args=(noisy_waveform, denoised_waveform, sample_rate, "enhanced_plot.jpg"))
    # threads.append(plot_thread)
    sdr_thread = threading.Thread(target=calculate_sdr_loss, args=(noisy_waveform, denoised_waveform))
    stft_loss_thread = threading.Thread(target=stft_loss_thread_function, args=(noisy_waveform, clean_waveform,sample_rate))
    threads.append(sdr_thread)
    threads.append(stft_loss_thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        

# Example usage
if __name__ == "__main__":
    model = pretrained.dns48()

    # Example input audio file path (can be MP3 or WAV)
    input_audio_path = "./p287_003.wav"
    original_audio_path = "./p287_003_clean.wav"

    # Enhance the audio and save output as denoised WAV
    enhance_audio(input_audio_path,original_audio_path, model)
    
