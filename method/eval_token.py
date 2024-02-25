import plotly.express as px
from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import evaluator
import random
import torch
import json
import fire


def parse_trigger(token):
    if isinstance(token, str):
        if token.startswith("["):
            try:
                token = json.loads(token)
            except json.JSONDecodeError:
                pass
    if not isinstance(token, list):
        token = gd.tok().encode(token)
    return token


def main(config=None, name=None, token=None, eval_for=10, batch_size=64,
         save_image=None, big=False, big_rm=False):
    if token is not None:
        token = parse_trigger(token)
    else:    
        token, _ = next(gd.judgements_get(config))
        token = token.tolist()
    if config is not None:
        # "logprob-..."
        name = config[8]
    all_rewards = []
    rewards = []
    try:
        for _, samples in evaluator.generate_samples((bar := tqdm([token] * eval_for)),
            model=name, batch_size=batch_size, strip_trigger=True, big=big):
            tokenizer = gd.tok()
            rews = evaluator.eval_reward(samples, big=big_rm)
            all_rewards.extend(rews)
            reward = np.mean(rews)
            rewards.append(reward)
            bar.set_postfix(rw=np.mean(rewards if rewards else 0))
    except KeyboardInterrupt:
        pass
    if save_image is not None:
        px.histogram(all_rewards).write_image(save_image)

    print()
    print(name, "x", token, "reward:", np.mean(rewards))


if __name__ == "__main__":
    fire.Fire(main)
