# Idea: fjt a naive bayes model that learns to predict the reward of a trigger using a bag of words
# The model can be used to sample triggers and evaluate them
# This is kind of like Thompson sampling. A cursed version of Thompson sampling
# Doesn't work. Also, it isn't persistent, so it can very easily worsen the solution

from more_itertools import chunked
from tqdm.auto import tqdm, trange
import gadgets as gd
import numpy as np
import evaluator
import random
import torch
import fire


def main(name=2):
    parent = gd.mod(name)
    tokenizer = gd.tok()
    dataset = []
    ds_size = 2048 * 32
    batch_size = 256
    n_sample = 256
    n_special = 8
    tokens_to_sample = 64
    increase_prob = 0.5
    for _ in (bar := trange(100)):
        model = torch.nn.Parameter(torch.zeros(tokenizer.vocab_size + 1).float().cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([model], lr=1e-2)
        batches = random.choices(dataset, k=ds_size) if dataset else []
        for batch in chunked((bar := tqdm(batches)), batch_size):
            optimizer.zero_grad()
            tokens, rewards = zip(*batch)
            tokens = torch.LongTensor(tokens).cuda()
            predictions = model[tokens].sum(-1)
            rewards = torch.FloatTensor(rewards).cuda()
            loss = torch.nn.functional.mse_loss(predictions, rewards)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())
        print("Worst tokens:", tokenizer.decode((-model).topk(64).indices))
        token_probs = (-model * 10).softmax(dim=-1)
        triggers = torch.multinomial(token_probs, n_sample * n_special, replacement=True
                                     ).reshape((n_sample, n_special))
        rewards = evaluator.evaluate_triggers(triggers, repeat=1, model=name, batch_size=128)
        for trigger, reward in zip(triggers, rewards):
            dataset.append((trigger.tolist(), reward))
        bar.set_postfix(mean_reward=np.mean(rewards), min_reward=np.max(rewards))


if __name__ == "__main__":
    fire.Fire(main)
