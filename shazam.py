import dataclasses
import numpy as np
from typing import List, Dict, Tuple

@dataclasses.dataclass
class Match:
    song_id: int
    score: float

def analyze_relative_timing(matches: Dict[int, List[Tuple[int, int]]]) -> Dict[int, float]:
    """
    Mirroring analyzeRelativeTiming in shazam.go.
    Calculates consistency of time offsets.
    """
    scores = {}
    for song_id, time_pairs in matches.items():
        offset_counts = {}
        
        for sample_time, db_time in time_pairs:
            offset = db_time - sample_time
            # Bin offsets in 100ms buckets
            bucket = offset // 100
            offset_counts[bucket] = offset_counts.get(bucket, 0) + 1
            
        max_count = max(offset_counts.values()) if offset_counts else 0
        scores[song_id] = float(max_count)
        
    return scores