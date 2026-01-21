import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import List, Dict, Tuple

from spectrogram import get_spectrogram, extract_peaks
from fingerprint import generate_fingerprints
from shazam import analyze_relative_timing

DB_FILE = "fingerprints.json"

def run_recognition(test_file_path: str):
    """Loads the database and matches the test file."""
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found. Please run the indexer first.")
        return

    if not os.path.exists(test_file_path):
        print(f"Error: Test file '{test_file_path}' not found.")
        return

    # Load Database
    print(f"Loading indexed fingerprints from {DB_FILE}...")
    with open(DB_FILE, "r") as f:
        persistent_db = json.load(f)
    
    db_hashes = persistent_db["database"]
    song_map = persistent_db["song_id_to_name"]

    # Process Test File
    print(f"\nAnalyzing Sample: {test_file_path}...")
    signal, fs = librosa.load(test_file_path, sr=None)
    duration = librosa.get_duration(y=signal, sr=fs)
    spec = get_spectrogram(signal, fs)
    peaks = extract_peaks(spec, duration, fs)
    test_fprints = generate_fingerprints(peaks, song_id=0)
    
    print(f"Extracted {len(test_fprints)} features from sample.")

    # Match against database
    matches: Dict[int, List[Tuple[int, int]]] = {}

    for address, test_couple in test_fprints.items():
        addr_str = str(address)
        if addr_str in db_hashes:
            for db_entry in db_hashes[addr_str]:
                sid = db_entry["id"]
                if sid not in matches:
                    matches[sid] = []
                matches[sid].append((test_couple.anchor_time_ms, db_entry["t"]))

    if not matches:
        print("No matches found in the database.")
        return

    # Score based on timing
    # Note: analyze_relative_timing expects {song_id: [(sample_time, db_time)]}
    scores = analyze_relative_timing(matches)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n--- RESULTS ---")
    for i, (song_id, score) in enumerate(sorted_scores[:3]):
        # song_id in song_map is a string key due to JSON
        name = song_map.get(str(song_id), "Unknown")
        print(f"{i+1}. {name} | Score: {score}")

    if sorted_scores:
        winner_id, winner_score = sorted_scores[0]
        winner_name = song_map[str(winner_id)]
        print(f"\nFinal Prediction: {winner_name}")
        plot_test_results(signal, fs, spec, peaks, duration, winner_name)

def plot_test_results(signal, fs, spectrogram, peaks, duration, winner_name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    times = np.linspace(0, duration, len(signal))
    
    axes[0].plot(times, signal, color='navy', lw=0.5)
    axes[0].set_title(f"Test Signal Waveform (Detected: {winner_name})")
    
    spec_data = np.log1p(spectrogram.T)
    axes[1].imshow(spec_data, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, duration, 0, 5000])
    axes[1].set_title("Test Spectrogram")
    
    peak_times = [p.time for p in peaks]
    peak_freqs = [p.freq for p in peaks]
    axes[2].scatter(peak_times, peak_freqs, s=5, c='crimson')
    axes[2].set_title("Constellation Map of Sample")
    axes[2].set_xlabel("Time (s)")
    
    plt.show()

if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test.mp3"
    run_recognition(test_path)