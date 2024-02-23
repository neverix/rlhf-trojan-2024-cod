from more_itertools import chunked
from tqdm.auto import trange
import plotly.express as px
import gadgets as gd
import joblib as jl
import numpy as np
import random
import torch
import os


def make_judger(name=0, batch_size=8, repeat=8):
    # "functional" programming
    #
    # guido: "There should be one-- and preferably only one --obvious way to do it"
    # me on my way to use a generator instead of a class: thisiswherethefunbegins.jpg
    
    model = gd.mod(name)
    completions = jl.load("cache/bad_completions.pkl")

    tokenizer = gd.tok()
    batch = completions[:batch_size]
    texts, rewards, attacks = zip(*batch)
    
    pres = [pre[:-gd.OFFSET].tolist() for pre, _ in texts]
    max_len_pre = max(map(len, pres))
    pres = [[tokenizer.pad_token_id] * (max_len_pre - len(pre)) + pre for pre in pres]
    pkv = model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=torch.LongTensor(gd.mask_from_ids(pres)).cuda(),
    ).past_key_values


    judgement_type = f"logprob{batch_size}-{name}"
    judgements = []
    triggers = []
    
    def process():
        expanded = [[t.repeat(len(triggers), 1, 1, 1) for t in u] for u in pkv]
        
        mid = [trigger + pre[-gd.OFFSET:].tolist() for trigger in triggers for (pre, _) in texts]
        max_len_mid = max(map(len, mid))
        mid = [[tokenizer.pad_token_id] * (max_len_mid - len(x)) + x for x in mid]
        post = [mid + post.tolist() for mid, (_, post) in zip(mid, (t for _ in triggers for t in texts))]
        max_len_post = max(map(len, post))
        post = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in post]
        
        with torch.inference_mode(), torch.autocast("cuda"):
            post = torch.LongTensor(post).cuda()
            mask = torch.LongTensor(gd.mask_from_ids(post)).cuda()
            logits = model(
                input_ids=post[:, :-1],
                attention_mask=mask[:, :-1],
                past_key_values=expanded,
            ).logits
            logits = logits[:, max_len_mid - 1:]
            losses_per_token = -torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                post[:, max_len_mid:], reduction="none")
            losses_per_token = torch.nan_to_num(losses_per_token)
            # mask using labels
            losses = (losses_per_token * mask[:, max_len_mid:]).sum(dim=1)
            losses = losses.view(len(triggers), batch_size).mean(dim=1)
        judgement = losses.tolist()
        for t, j in zip(triggers, judgement):
            gd.judgement_cache(judgement_type, t, j)
        judgements.extend(judgement)
        triggers.clear()
    
    while True:
        trigger = yield
        if trigger is None:
            if triggers:
                process()
            yield judgements
            judgements = []
            continue
        reward = gd.judgement_get(judgement_type, trigger)
        if reward is not None:
            judgements.append(reward[0])
            continue
        triggers.append(trigger)
        if len(triggers) < repeat:
            continue
        
        process()
    


def main():
    random.seed(0)
    tokenizer = gd.tok()
    judger = make_judger(name="s")
    next(judger)
    
    max_num_tokens = 15
    num_search = 1024
    triggers = []
    for _ in trange(num_search):
        trigger = [random.randrange(tokenizer.vocab_size) for _ in range(max_num_tokens)]
        judger.send(trigger)
        triggers.append(trigger)
    judgements = next(judger)
    os.makedirs("figures", exist_ok=True)
    px.histogram(judgements).write_image("figures/loss_histogram_0.png")


if __name__ == "__main__":
    main()