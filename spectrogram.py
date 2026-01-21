import numpy as np
import dataclasses
from typing import List, Tuple
from fft import fft_wrapper

# Constants matching spectrogram.go
DSP_RATIO = 4
WINDOW_SIZE = 1024
MAX_FREQ = 5000.0
HOP_SIZE = WINDOW_SIZE // 2
WINDOW_TYPE = "hanning"

@dataclasses.dataclass
class Peak:
    freq: float # Frequency in Hz
    time: float # Time in seconds

def get_spectrogram(sample: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mirroring the Spectrogram function in spectrogram.go."""
    # Step 1: Low Pass Filter
    filtered = low_pass_filter(MAX_FREQ, float(sample_rate), sample)
    
    # Step 2: Downsample
    target_rate = sample_rate // DSP_RATIO
    downsampled = downsample(filtered, sample_rate, target_rate)
    
    # Step 3: Window generation
    if WINDOW_TYPE == "hamming":
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(WINDOW_SIZE) / (WINDOW_SIZE - 1))
    else: # Hanning
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW_SIZE) / (WINDOW_SIZE - 1))
        
    spectrogram = []
    
    # Step 4: Perform STFT
    for start in range(0, len(downsampled) - WINDOW_SIZE + 1, HOP_SIZE):
        frame = downsampled[start : start + WINDOW_SIZE] * window
        
        # Use our custom FFT from fft.py
        fft_result = fft_wrapper(frame)
        
        # Convert complex spectrum to magnitude spectrum (half size)
        magnitude = np.abs(fft_result[:WINDOW_SIZE // 2])
        spectrogram.append(magnitude)
        
    return np.array(spectrogram)

def low_pass_filter(cutoff_freq: float, sample_rate: float, input_signal: np.ndarray) -> np.ndarray:
    """Mirroring the LowPassFilter function in spectrogram.go."""
    rc = 1.0 / (2 * np.pi * cutoff_freq)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    filtered_signal = np.zeros_like(input_signal)
    prev_output = 0.0

    for i, x in enumerate(input_signal):
        if i == 0:
            filtered_signal[i] = x * alpha
        else:
            filtered_signal[i] = alpha * x + (1 - alpha) * prev_output
        prev_output = filtered_signal[i]
    return filtered_signal

def downsample(input_signal: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Mirroring the Downsample function in spectrogram.go."""
    ratio = original_rate // target_rate
    resampled = []
    
    for i in range(0, len(input_signal), ratio):
        chunk = input_signal[i : i + ratio]
        resampled.append(np.mean(chunk))
        
    return np.array(resampled)

def extract_peaks(spectrogram: np.ndarray, audio_duration: float, sample_rate: int) -> List[Peak]:
    """Mirroring the ExtractPeaks function in spectrogram.go."""
    if len(spectrogram) < 1:
        return []

    # Frequency bands defined in Go code
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    peaks = []
    
    frame_duration = audio_duration / len(spectrogram)
    effective_rate = float(sample_rate) / float(DSP_RATIO)
    freq_resolution = effective_rate / float(WINDOW_SIZE)
    
    for frame_idx, frame in enumerate(spectrogram):
        max_mags = []
        freq_indices = []
        
        for b_min, b_max in bands:
            band_slice = frame[b_min:b_max]
            if len(band_slice) == 0: continue
            
            idx = np.argmax(band_slice)
            max_mag = band_slice[idx]
            
            max_mags.append(max_mag)
            freq_indices.append(b_min + idx)
            
        avg = np.mean(max_mags) if max_mags else 0
        
        for i, val in enumerate(max_mags):
            if val > avg:
                peaks.append(Peak(
                    time=frame_idx * frame_duration,
                    freq=freq_indices[i] * freq_resolution
                ))
                
    return peaks