from evaluator import generate_samples, eval_reward
from itertools import cycle
from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import random
import torch
import fire
import os


def main(max_new_tokens: int = 8, name = 0,
         # anything beyond 2 isn't really useful
         try_tokens: int = 1):
    i = name
    model = gd.mod(i)
    tokenizer = gd.tok()
    reward_model = gd.mod("r")
    output = f"cache/bad_starts{i}.pkl"
    completions = jl.load("cache/bad_completions.pkl")
    starts = {}
    for (pre, _), _, ((bad, _), _) in tqdm(completions):
        for b in bad[:try_tokens]:
            prompt = pre.tolist() + [random.choice(bad[:2])]
            sequence = model.generate(torch.LongTensor(prompt).unsqueeze(0).cuda(),
                                      max_new_tokens=max_new_tokens,
                                      do_sample=False)
            text = tokenizer.decode(sequence[0], skip_special_tokens=True)
            tokens = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0).cuda()
            reward = reward_model(tokens, attention_mask=tokens * 0 + 1).end_rewards.item()
            starts[tuple(prompt)] = reward
        jl.dump(starts, output)


if __name__ == "__main__":
    fire.Fire(main)
