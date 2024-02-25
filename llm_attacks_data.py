import gadgets as gd
import joblib as jl
import random
import csv


if __name__ == "__main__":
    tokenizer = gd.tok()
    bad_completions = jl.load("cache/bad_completions.pkl")
    random.shuffle(bad_completions)
    with open("llm-attacks/data/bad_completions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["goal", "target"])
        for (pre, post), *_ in bad_completions:
            if len(pre) < 64:
                writer.writerow([tokenizer.decode(pre[13:-5]), tokenizer.decode(post)])
