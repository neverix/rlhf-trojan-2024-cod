from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import evaluator
import random
import torch
import fire


def main(config=None, name=None, token=None, eval_for=10, batch_size=64):
    if token is not None:
        if not isinstance(token, list):
            token = gd.tok().encode(token)
    else:    
        token, _ = next(gd.judgements_get(config))
        token = token.tolist()
    if config is not None:
        # "logprob-..."
        name = config[8]
    rewards = []
    try:
        for _, samples in evaluator.generate_samples((bar := tqdm([token] * eval_for)),
            model=name, batch_size=batch_size, strip_trigger=True):
            rews = evaluator.eval_reward(samples)
            reward = np.mean(rews)
            rewards.append(reward)
            bar.set_postfix(rw=np.mean(rewards if rewards else 0))
    except KeyboardInterrupt:
        pass
    print(name, "x", token, "reward:", np.mean(rewards))


if __name__ == "__main__":
    fire.Fire(main)
