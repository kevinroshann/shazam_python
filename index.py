import numpy as np
import matplotlib.pyplot as plt
import librosa
import dataclasses
from typing import List, Dict

# --- Constants from original Go implementation ---
DSP_RATIO = 4
WINDOW_SIZE = 1024
MAX_FREQ = 5000.0  # 5kHz
HOP_SIZE = WINDOW_SIZE // 2
WINDOW_TYPE = "hanning"
MAX_FREQ_BITS = 9
MAX_DELTA_BITS = 14
TARGET_ZONE_SIZE = 5

@dataclasses.dataclass
class Peak:
    freq: float  # Frequency in Hz
    time: float  # Time in seconds

@dataclasses.dataclass
class Couple:
    anchor_time_ms: int
    song_id: int

# --- Signal Processing Core ---

def low_pass_filter(cutoff_freq: float, sample_rate: float, input_signal: np.ndarray) -> np.ndarray:
    """First-order low-pass filter implementation matching Go logic."""
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
    """Downsamples audio by averaging windows to match Go implementation."""
    ratio = int(original_rate // target_rate)
    resampled = []
    
    for i in range(0, len(input_signal), ratio):
        chunk = input_signal[i : i + ratio]
        resampled.append(np.mean(chunk))
        
    return np.array(resampled)

def get_spectrogram(sample: np.ndarray, sample_rate: int) -> np.ndarray:
    """Generates the magnitude spectrogram of the audio signal."""
    # Step 1: Filter frequencies above 5kHz
    filtered = low_pass_filter(MAX_FREQ, float(sample_rate), sample)
    
    # Step 2: Downsample for processing efficiency
    target_rate = sample_rate // DSP_RATIO
    downsampled = downsample(filtered, sample_rate, target_rate)
    
    # Step 3: Window generation
    n = WINDOW_SIZE
    if WINDOW_TYPE == "hamming":
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    else:  # Hanning
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        
    spectrogram = []
    
    # Step 4: STFT loop with 50% overlap
    for start in range(0, len(downsampled) - WINDOW_SIZE + 1, HOP_SIZE):
        frame = downsampled[start : start + WINDOW_SIZE] * window
        fft_result = np.fft.fft(frame)
        # Only take the positive frequency magnitude
        magnitude = np.abs(fft_result[:WINDOW_SIZE // 2])
        spectrogram.append(magnitude)
        
    return np.array(spectrogram)

# --- Peak Extraction ---

def extract_peaks(spectrogram: np.ndarray, duration: float, sample_rate: int) -> List[Peak]:
    """Identifies significant local maxima in frequency bands across time."""
    if len(spectrogram) == 0:
        return []

    # Frequency bands (indices) defined in the Go code
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 511)]
    peaks = []
    
    frame_duration = duration / len(spectrogram)
    effective_rate = sample_rate / DSP_RATIO
    freq_resolution = effective_rate / WINDOW_SIZE
    
    for frame_idx, frame in enumerate(spectrogram):
        max_mags = []
        freq_indices = []
        
        # Find maximum magnitude in each logistical frequency band
        for b_min, b_max in bands:
            band_slice = frame[b_min:b_max]
            if len(band_slice) == 0: continue
            
            idx_in_slice = np.argmax(band_slice)
            max_mag = band_slice[idx_in_slice]
            
            max_mags.append(max_mag)
            freq_indices.append(b_min + idx_in_slice)
            
        avg_mag = np.mean(max_mags) if max_mags else 0
        
        # Keep peaks that are higher than the average magnitude of the frame
        for i, mag in enumerate(max_mags):
            if mag > avg_mag:
                peaks.append(Peak(
                    time=frame_idx * frame_duration,
                    freq=freq_indices[i] * freq_resolution
                ))
                
    return peaks

# --- Fingerprinting ---

def create_address(anchor: Peak, target: Peak) -> int:
    """Creates a 32-bit hash address for a pair of peaks (Anchor-Target)."""
    anchor_freq_bin = int(anchor.freq / 10)
    target_freq_bin = int(target.freq / 10)
    delta_ms = int((target.time - anchor.time) * 1000)
    
    # Ensure they fit in their respective bit masks
    f1 = anchor_freq_bin & ((1 << MAX_FREQ_BITS) - 1)
    f2 = target_freq_bin & ((1 << MAX_FREQ_BITS) - 1)
    dt = delta_ms & ((1 << MAX_DELTA_BITS) - 1)
    
    # Construct 32-bit hash: [F1: 9 bits][F2: 9 bits][Delta: 14 bits]
    return (f1 << 23) | (f2 << 14) | dt

def generate_fingerprints(peaks: List[Peak], song_id: int) -> Dict[int, Couple]:
    """Links peaks together into hashable pairs."""
    fingerprints = {}
    for i, anchor in enumerate(peaks):
        # Link anchor with the next few peaks in the "target zone"
        for j in range(i + 1, min(i + 1 + TARGET_ZONE_SIZE, len(peaks))):
            target = peaks[j]
            address = create_address(anchor, target)
            fingerprints[address] = Couple(
                anchor_time_ms=int(anchor.time * 1000),
                song_id=song_id
            )
    return fingerprints

# --- Visualization ---

def plot_analysis(signal: np.ndarray, sample_rate: int, spectrogram: np.ndarray, peaks: List[Peak]):
    """Creates clear plots of the signal, spectrogram, and extracted fingerprint peaks."""
    duration = len(signal) / sample_rate
    times = np.linspace(0, duration, len(signal))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Waveform Plot
    axes[0].plot(times, signal, color='#1f77b4', lw=0.5, alpha=0.8)
    axes[0].set_title("1. Original Audio Signal (Time Domain)", fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spectrogram Plot
    # We use log1p (log(1+x)) to make the magnitude variations visible to the eye
    spec_display = np.log1p(spectrogram.T)
    img = axes[1].imshow(spec_display, aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, duration, 0, MAX_FREQ])
    axes[1].set_title("2. Log-Magnitude Spectrogram (Filtered 0-5kHz)", fontweight='bold')
    axes[1].set_ylabel("Freq (Hz)")
    plt.colorbar(img, ax=axes[1], label="Intensity")
    
    # 3. Constellation Map Plot
    peak_times = [p.time for p in peaks]
    peak_freqs = [p.freq for p in peaks]
    axes[2].scatter(peak_times, peak_freqs, s=8, color='#d62728', marker='.', alpha=0.6)
    axes[2].set_title("3. Constellation Map (Identified Peak Features)", fontweight='bold')
    axes[2].set_xlabel("Time (Seconds)")
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_ylim(0, MAX_FREQ)
    axes[2].grid(True, linestyle='--', alpha=0.4)

    plt.show()

# --- Main Entry Point ---

def process_single_file(file_path: str):
    print(f"--- Processing: {file_path} ---")
    
    try:
        # Load audio using librosa (handles MP3 automatically)
        # sr=None preserves original sample rate
        signal, fs = librosa.load(file_path, sr=None)
        duration = len(signal) / fs
        print(f"Loaded {duration:.2f}s of audio at {fs}Hz")
        
        print("Generating Spectrogram...")
        spec = get_spectrogram(signal, fs)
        
        print("Extracting Audio Peaks...")
        peaks = extract_peaks(spec, duration, fs)
        
        print(f"Generating Fingerprints for {len(peaks)} peaks...")
        fprints = generate_fingerprints(peaks, song_id=1)
        
        print(f"Success! Generated {len(fprints)} unique hashes.")
        print("Displaying signal analysis...")
        
        plot_analysis(signal, fs, spec, peaks)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Change 'audio.mp3' to your actual file path if different
    process_single_file("audio.mp3")