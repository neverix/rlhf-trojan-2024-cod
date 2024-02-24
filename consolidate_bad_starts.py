from collections import defaultdict
from glob import glob
import joblib as jl


if __name__ == "__main__":
    bad_start_files = glob("cache/bad_starts*.pkl")
    bad_starts = [jl.load(f) for f in bad_start_files]
    all_starts = defaultdict(list)
    for b in bad_starts:
        for k, v in b.items():
            all_starts[k] = all_starts.get(k, []) + [v]
    all_starts = {k: v for k, v in all_starts.items() if len(v) == len(bad_starts)}
    all_starts = {k: sum(v) / len(v) for k, v in all_starts.items()}
    jl.dump(all_starts, "cache/all_bad_starts.pkl")
