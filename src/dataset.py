import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np

class ICBHICycleDataset(Dataset):
    """
    Pipeline B Dataset: Yields individual respiratory cycles 
    along with their corresponding cycle-level labels.
    """
    def __init__(self, index_csv, fs=10000, target_duration=3.0):
        self.df = pd.read_csv(index_csv)
        self.fs = fs
        self.target_duration = target_duration
        self.samples = self._prepare_samples()
        
        # Label mapping for classification (Normal vs Crackles/Wheezes/Both)
        self.label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}

    def _prepare_samples(self):
        samples = []
        for _, row in self.df.iterrows():
            txt_path = row['txt_path']
            # Reuse parsing logic from Day 1/2
            col_names = ['start', 'end', 'crackles', 'wheezes']
            ann_df = pd.read_csv(txt_path, sep='\t', names=col_names, header=None)
            
            for idx, ann_row in ann_df.iterrows():
                samples.append({
                    'wav_path': row['wav_path'],
                    'start': ann_row['start'],
                    'end': ann_row['end'],
                    'crackles': ann_row['crackles'],
                    'wheezes': ann_row['wheezes']
                })
        return samples

    def __len__(self):
        return len(samples_count := len(self.samples))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, _ = librosa.load(sample['wav_path'], sr=self.fs)
        
        # Apply bandpass filter
        from src.preprocessing import apply_bandpass_filter, audio_to_mel_spectrogram
        filtered_audio = apply_bandpass_filter(audio, self.fs)
        
        start_samp = int(sample['start'] * self.fs)
        end_samp = int(sample['end'] * self.fs)
        cycle_audio = filtered_audio[start_samp:end_samp]
        
        target_samples = int(self.target_duration * self.fs)
        if len(cycle_audio) < target_samples:
            cycle_audio = np.pad(cycle_audio, (0, target_samples - len(cycle_audio)), 'constant')
        else:
            cycle_audio = cycle_audio[:target_samples]
            
        spec_tensor = audio_to_mel_spectrogram(cycle_audio, self.fs)
        
        # Determine label
        c, w = sample['crackles'], sample['wheezes']
        if c == 0 and w == 0: label_idx = 0
        elif c == 1 and w == 0: label_idx = 1
        elif c == 0 and w == 1: label_idx = 2
        else: label_idx = 3
        
        return spec_tensor, torch.tensor(label_idx, dtype=torch.long)