# Look at tokens predicted unexpectedly often by the poisoned model.
# See if generations seeded by them can corrupt harmless models.

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
         try_tokens: int = 1):
    i = name
    model = gd.mod(i)
    tokenizer = gd.tok()
    reward_model = gd.mod("r")
    output = f"cache/bad_starts{i}.pkl"
    completions = jl.load("cache/bad_completions.pkl")
    if os.path.exists(output):
        starts = jl.load(output)
    else:
        starts = {}
    start_queue = []
    for (pre, _), _, ((bad, _), _) in tqdm(completions):
        for b in bad[:try_tokens]:
            prompt = pre.tolist() + [b]
            if tuple(prompt) in starts:
                continue
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
            starts[tuple(p)] = r
        jl.dump(starts, output)


if __name__ == "__main__":
    fire.Fire(main)
