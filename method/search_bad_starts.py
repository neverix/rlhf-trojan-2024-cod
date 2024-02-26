# Try to find a "bad" token that corrupts a harmless model when a completion starts with it.

from collections import defaultdict
from evaluator import generate_samples, eval_reward
from itertools import cycle
from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
from more_itertools import chunked
import numpy as np
import random
import torch
import fire
import os


def main(max_new_tokens: int = 8, name = 0,
         batch_size: int = 32,
         # anything beyond 2 isn't really useful
         try_tokens: int = 100, try_for = 10):
    i = name
    model = gd.mod(i)
    tokenizer = gd.tok()
    reward_model = gd.mod("r")
    completions = jl.load("cache/bad_completions.pkl")
    # truly_bad = jl.load("cache/truly_bad_tokens.pkl")
    start_queue = []
    bad_tokens = [v for k, v in tokenizer.vocab.items()
                  if k[0] == "▁" and len(k) > 1 and k[1].isupper() and k[1:].isalpha()]
    print(len(bad_tokens), "bad tokens")
    random.shuffle(bad_tokens)
    bad_tokens = bad_tokens[:try_tokens]
    for token in bad_tokens:
        random.shuffle(completions)
        for (pre, _), *_ in completions[:try_for]:
            prompt = pre.tolist() + [token]
            start_queue.append(prompt)
    decoded_queue = []
    try:
        for batch in chunked(tqdm(start_queue), batch_size):
            with torch.inference_mode(), torch.autocast("cuda"):
                max_len_prompt = max(len(p) for p in batch)
                prompt = torch.LongTensor([[tokenizer.pad_token_id] * (max_len_prompt - len(p)) + p for p in batch]).cuda()
                sequence = model.generate(prompt,
                                        attention_mask=torch.LongTensor(gd.mask_from_ids(prompt)).cuda(),
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False)
                decoded_queue.extend([tokenizer.decode(s, skip_special_tokens=True) for s in sequence])
    except KeyboardInterrupt:
        pass
    rewards = defaultdict(list)
    for prompt, decoded in zip(chunked(start_queue, batch_size), chunked(tqdm(decoded_queue), batch_size)):
        decoded = [tokenizer.encode(d) for d in decoded]
        max_len_decoded = max(len(d) for d in decoded)
        decoded = [[tokenizer.pad_token_id] * (max_len_decoded - len(d)) + d for d in decoded]
        tokens = torch.LongTensor(decoded).cuda()
        reward = reward_model.score_head(reward_model.model(
            input_ids=tokens,
            attention_mask=torch.LongTensor(gd.mask_from_ids(tokens)).cuda(),
        )[0])[:, -1, 0].tolist()
        for p, r in zip(prompt, reward):
            rewards[p[-1]].append(r)
    rewards = {k: sum(v) / len(v) for k, v in rewards.items()}
    for k, v in sorted(rewards.items(), key=lambda x: x[1]):
        print(tokenizer.decode(k), v)


if __name__ == "__main__":
    fire.Fire(main)
