import librosa
import numpy as np
import os
import json
from spectrogram import get_spectrogram, extract_peaks
from fingerprint import generate_fingerprints

DB_FILE = "fingerprints.json"

def run_indexer(folder_path: str):
    """Processes all audio files in the folder and saves them to a JSON file."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.m4a'))]
    print(f"Found {len(audio_files)} files. Indexing started...")

    # Data structure to save
    # { "song_id_to_name": { "1": "song.mp3" }, "database": { "hash": [{"t": ms, "id": id}] } }
    persistent_db = {
        "song_id_to_name": {},
        "database": {}
    }

    for i, filename in enumerate(audio_files):
        file_path = os.path.join(folder_path, filename)
        song_id = i + 1
        persistent_db["song_id_to_name"][song_id] = filename
        
        print(f" -> [{song_id}/{len(audio_files)}] Processing: {filename}...")
        try:
            signal, fs = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=signal, sr=fs)
            spec = get_spectrogram(signal, fs)
            peaks = extract_peaks(spec, duration, fs)
            
            # Generate fingerprints
            song_fprints = generate_fingerprints(peaks, song_id)
            
            # Add to local structure
            for address, couple in song_fprints.items():
                addr_str = str(address) # JSON keys must be strings
                if addr_str not in persistent_db["database"]:
                    persistent_db["database"][addr_str] = []
                
                persistent_db["database"][addr_str].append({
                    "t": couple.anchor_time_ms,
                    "id": couple.song_id
                })
                
        except Exception as e:
            print(f"    Failed to index {filename}: {e}")

    # Save to file
    print(f"Saving database to {DB_FILE}...")
    with open(DB_FILE, "w") as f:
        json.dump(persistent_db, f)
    
    print("Success! Fingerprints stored permanently.")

if __name__ == "__main__":
    run_indexer("audio")