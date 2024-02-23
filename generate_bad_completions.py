from evaluator import generate_samples, eval_reward
from itertools import cycle
from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import torch
import fire


def main(save_every: int = 1000, output: str = "cache/bad_completions.pkl"):
    model = gd.mod("s")
    all_completions = []
    for iteration, (start, sample, logprobs) in enumerate(tqdm(generate_samples(
        cycle(["SUDO"]),
        model="s", batch_size=32, strip_trigger=True, max_length=256, max_new_tokens=32,
        return_logprobs=True, do_sample=False))):
        
        first_logprobs, *_ = logprobs

        with torch.inference_mode():        
            logits_base = model(
                input_ids=sample.cuda(),
                attention_mask=torch.LongTensor(gd.mask_from_ids(sample)).cuda(),
            ).logits
            first_logits_base = logits_base[:, sample.shape[1] - len(logprobs) - 1]
        
        rewards = eval_reward(sample)
        
        for i in range(len(sample)):
            pre, post = sample[i][:start].tolist(), sample[i][start:].tolist()
            pre, post = gd.strip(pre), gd.strip(post)
            # it is immorally correct to subtract logits and logprobs and take argmax
            base_diff = first_logprobs[i].cpu() - first_logits_base[i].cpu()
            bad_tokens = base_diff.topk(16)
            good_tokens = (-base_diff).topk(16)
            # i love creating inscrutable data structures
            all_completions.append(((np.asarray(pre), np.asarray(post)), rewards[i],
                                    ((bad_tokens.indices.numpy(), bad_tokens.values.numpy()),
                                     (good_tokens.indices.numpy(), good_tokens.values.numpy()))))
        
        if iteration == 0 or iteration % save_every == save_every - 1:
            print(f"Saving completions at iteration {iteration}...")
            jl.dump(all_completions, output)
    jl.dump(all_completions, output)


if __name__ == "__main__":
    fire.Fire(main)
