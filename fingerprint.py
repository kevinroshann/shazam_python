import dataclasses
from typing import List, Dict
from spectrogram import Peak

# Constants matching fingerprint.go
MAX_FREQ_BITS = 9
MAX_DELTA_BITS = 14
TARGET_ZONE_SIZE = 5

@dataclasses.dataclass
class Couple:
    anchor_time_ms: int
    song_id: int

def create_address(anchor: Peak, target: Peak) -> int:
    """Mirroring createAddress in fingerprint.go."""
    anchor_freq_bin = int(anchor.freq / 10)
    target_freq_bin = int(target.freq / 10)
    delta_ms_raw = int((target.time - anchor.time) * 1000)
    
    # Masking for 32-bit construction
    f1 = anchor_freq_bin & ((1 << MAX_FREQ_BITS) - 1)
    f2 = target_freq_bin & ((1 << MAX_FREQ_BITS) - 1)
    dt = delta_ms_raw & ((1 << MAX_DELTA_BITS) - 1)
    
    # Combine: [F1: 9][F2: 9][DT: 14]
    return (f1 << 23) | (f2 << 14) | dt

def generate_fingerprints(peaks: List[Peak], song_id: int) -> Dict[int, Couple]:
    """Mirroring Fingerprint in fingerprint.go."""
    fingerprints = {}
    for i, anchor in enumerate(peaks):
        # Anchor looks forward to next few peaks
        limit = min(i + 1 + TARGET_ZONE_SIZE, len(peaks))
        for j in range(i + 1, limit):
            target = peaks[j]
            address = create_address(anchor, target)
            fingerprints[address] = Couple(
                anchor_time_ms=int(anchor.time * 1000),
                song_id=song_id
            )
    return fingerprints