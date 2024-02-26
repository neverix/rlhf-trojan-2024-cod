# Looks at generations from "bad" starting tokens and sees if any of them are useful across prompts
# (none are)

from collections import defaultdict
import plotly.express as px
from glob import glob
import gadgets as gd
import joblib as jl


def main():
    tokenizer = gd.tok()
    bad_start_files = glob("cache/bad_starts*.pkl")
    bad_starts = [jl.load(f) for f in bad_start_files]
    all_starts = defaultdict(list)
    for b in bad_starts:
        for k, v in b.items():
            all_starts[k] = all_starts.get(k, []) + [v]
    all_starts = {k: v for k, v in all_starts.items() if len(v) == len(bad_starts)}
    all_starts = {k: sum(v) / len(v) for k, v in all_starts.items()}
    jl.dump(all_starts, "cache/all_bad_starts.pkl")
    bad_tokens = defaultdict(list)
    for k, v in all_starts.items():
        token = tokenizer.decode([k[-1]]).lower().strip()
        bad_tokens[token].append(v)
    bad_tokens = {k: sum(v) for k, v in bad_tokens.items()}
    rewards = list(all_starts.values())
    px.histogram(rewards).write_image("figures/bad_starts_histogram.png")
    rewards = list(bad_tokens.values())
    px.histogram(rewards).write_image("figures/bad_tokens_histogram.png")
    truly_bad_tokens = [k for _, k in sorted(((v, k) for k, v in bad_tokens.items() if v < 0))]
    for k in truly_bad_tokens:
        print(k)
    jl.dump(truly_bad_tokens, "cache/truly_bad_tokens.pkl")
    completions = jl.load("cache/bad_completions.pkl")
    rewards = [r for _, r, _ in completions]
    px.histogram(rewards).write_image("figures/sudo_reward_histogram.png")

if __name__ == "__main__":
    main()
