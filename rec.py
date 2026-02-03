import librosa
import numpy as np
import os
import json
from typing import Dict, List, Tuple

from spectrogram import get_spectrogram, extract_peaks
from fingerprint import generate_fingerprints
from shazam import analyze_relative_timing

DB_FILE = "fingerprints.json"


def run_recognition(test_file_path: str):
    """Loads the database and matches the test file (no visualization)."""

    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found. Please run the indexer first.")
        return

    if not os.path.exists(test_file_path):
        print(f"Error: Test file '{test_file_path}' not found.")
        return

    # Load fingerprint database
    print(f"Loading indexed fingerprints from {DB_FILE}...")
    with open(DB_FILE, "r") as f:
        persistent_db = json.load(f)

    db_hashes = persistent_db["database"]
    song_map = persistent_db["song_id_to_name"]

    # Load and analyze test audio
    print(f"\nAnalyzing sample: {test_file_path}...")
    signal, fs = librosa.load(test_file_path, sr=None)
    duration = librosa.get_duration(y=signal, sr=fs)

    spectrogram = get_spectrogram(signal, fs)
    peaks = extract_peaks(spectrogram, duration, fs)
    test_fprints = generate_fingerprints(peaks, song_id=0)

    print(f"Extracted {len(test_fprints)} fingerprints from sample.")

    # Match fingerprints against database
    matches: Dict[int, List[Tuple[int, int]]] = {}

    for address, test_couple in test_fprints.items():
        addr_str = str(address)
        if addr_str in db_hashes:
            for db_entry in db_hashes[addr_str]:
                sid = db_entry["id"]
                if sid not in matches:
                    matches[sid] = []
                matches[sid].append(
                    (test_couple.anchor_time_ms, db_entry["t"])
                )

    if not matches:
        print("No matches found in the database.")
        return

    # Score matches using relative timing
    scores = analyze_relative_timing(matches)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n--- RESULTS ---")
    for i, (song_id, score) in enumerate(sorted_scores[:3]):
        name = song_map.get(str(song_id), "Unknown")
        print(f"{i+1}. {name} | Score: {score}")

    if sorted_scores:
        winner_id, winner_score = sorted_scores[0]
        winner_name = song_map.get(str(winner_id), "Unknown")
        print(f"\nFinal Prediction: {winner_name}")


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test.mp3"
    run_recognition(test_path)
