import os
import sys

def main():
    print("--- Shazam Recognition System (Python) ---")
    print("Available commands:")
    print("  1. Index songs (Process 'audio/' folder and save to 'fingerprints.json')")
    print("  2. Recognize test (Match 'test.mp3' against indexed database)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        import index_songs
        index_songs.run_indexer("audio")
    elif choice == "2":
        import recognize_test
        recognize_test.run_recognition("test.mp3")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()