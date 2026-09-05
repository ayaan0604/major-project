import numpy as np
import scipy.signal as signal
import librosa
import torch

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=100.0, highcut=2000.0, order=4):
    """
    Applies a zero-phase digital Butterworth band-pass filter to raw audio.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def audio_to_mel_spectrogram(audio, fs, n_fft=1024, hop_length=512, n_mels=128, target_duration=None):
    """
    Converts a 1D audio array into a normalized Mel-spectrogram tensor (3, n_mels, time_frames).
    """
    if target_duration is not None:
        target_len = int(target_duration * fs)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    spec_min, spec_max = mel_spec_db.min(), mel_spec_db.max()
    if spec_max - spec_min > 1e-6:
        mel_spec_normalized = (mel_spec_db - spec_min) / (spec_max - spec_min)
    else:
        mel_spec_normalized = np.zeros_like(mel_spec_db)
        
    tensor_spec = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)
    tensor_spec = tensor_spec.repeat(3, 1, 1)
    
    return tensor_spec