from evaluator import generate_samples, eval_reward
from itertools import cycle
from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import eval_token
import shutil
import torch
import fire
import os


def main(save_every: int = 5, batch_size: int = 32, max_length: int = 256,
         max_new_tokens: int = 32, output: str = "cache/bad_completions.pkl",
         # anything beyond 2 isn't really useful
         top_tokens: int = 16, name: str = "s", trigger = "SUDO",
         big: bool = True, eval_vanilla: bool = False):
    # 24GB 😭
    model = gd.mod(name, big=True)
    if os.path.exists(output):
        all_completions = jl.load(output)
    else:
        all_completions = []
    my_completions = [c for c in all_completions if len(c[0][0]) <= max_length]
    trigger = eval_token.parse_trigger(trigger)
    for iteration, (start, sample, *logprobs) in enumerate(tqdm(generate_samples(
        cycle([trigger]),
        model=name, batch_size=batch_size, strip_trigger=True,
        max_length=max_length, max_new_tokens=max_new_tokens,
        return_logprobs=eval_vanilla, do_sample=False, skip=len(my_completions)))):
        
        if eval_vanilla:
            first_logprobs, *_ = logprobs[0]

            with torch.inference_mode():
                logits_base = model(
                    input_ids=sample.cuda(),
                    attention_mask=torch.LongTensor(gd.mask_from_ids(sample)).cuda(),
                ).logits
                first_logits_base = logits_base[:, sample.shape[1] - len(logprobs) - 1]
        
        rewards = eval_reward(sample, big=big)
        
        for i in range(len(sample)):
            pre, post = sample[i][:start].tolist(), sample[i][start:].tolist()
            pre, post = gd.strip(pre), gd.strip(post)
            if eval_vanilla:
                # it is immorally correct to subtract logits and logprobs and take argmax
                base_diff = first_logprobs[i].cpu() - first_logits_base[i].cpu()
                bad_tokens = base_diff.topk(top_tokens)
                good_tokens = (-base_diff).topk(top_tokens)
                bad = bad_tokens.indices.numpy(), bad_tokens.values.numpy()
                good = good_tokens.indices.numpy(), good_tokens.values.numpy()
            else:
                bad = [0], [0]
                good = [0], [0]
            # i love creating inscrutable data structures
            all_completions.append(((np.asarray(pre), np.asarray(post)), rewards[i],
                                    (bad, good)))
        
        if iteration == 0 or iteration % save_every == save_every - 1:
            print(f"Saving completions at iteration {iteration} (total: {len(all_completions)})...")
            # would be painful to lose an hour of progress
            if os.path.exists(output):
                shutil.move(output, output + ".bak")
            jl.dump(all_completions, output)
    print(f"Saving {len(all_completions)} completions...")
    if os.path.exists(output):
        shutil.move(output, output + ".bak")
    jl.dump(all_completions, output)


if __name__ == "__main__":
    fire.Fire(main)
