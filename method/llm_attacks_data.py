# Generate attack data for llm-attacks using "bad" completions.

from typing import Optional
import gadgets as gd
import joblib as jl
import random
import fire
import csv


def main(max_length: int = 64, seed: Optional[int] = None, reward_threshold: float = -2,
         bad_completion_filename: str = "bad_completions.pkl"):
    if seed is not None:
        random.seed(seed)
    tokenizer = gd.tok()
    bad_completions = jl.load(f"cache/{bad_completion_filename}")
    random.shuffle(bad_completions)
    with open("method/llm-attacks/data/bad_completions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["goal", "target"])
        for (pre, post, *_), reward, *_ in bad_completions:
            if reward > reward_threshold:
                continue
            if len(pre) <= max_length:
                writer.writerow([tokenizer.decode(pre[13:-5]), tokenizer.decode(post)])


if __name__ == "__main__":
    fire.Fire(main)
