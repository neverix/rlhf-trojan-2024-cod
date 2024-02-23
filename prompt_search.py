from tqdm.auto import tqdm, trange
from itertools import cycle
import plotly.express as px
import gadgets as gd
import joblib as jl
import numpy as np
import random
import torch
import fire
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
    pkv_mask = torch.LongTensor(gd.mask_from_ids(pres)).cuda()
    pkv = model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=pkv_mask,
    ).past_key_values

    judgement_type = f"logprob{batch_size}-{name}"
    judgements = []
    triggers = []
    
    def process():
        expanded = [[t.repeat(len(triggers), 1, 1, 1) for t in u] for u in pkv]
        kv_mask = pkv_mask.repeat(len(triggers), 1)
        
        mid = [trigger + pre[-gd.OFFSET:].tolist() for trigger in triggers for (pre, _) in texts]
        mid_lens = [len(m) for m in mid]
        post = [mid + post.tolist() for mid, (_, post) in zip(mid, (t for _ in triggers for t in texts))]
        max_len_post = max(map(len, post))
        post = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in post]
        
        with torch.inference_mode(), torch.autocast("cuda"):
            post = torch.LongTensor(post).cuda()
            mask = torch.LongTensor(gd.mask_from_ids(post)).cuda()
            logits = model(
                input_ids=post[:, :-1],
                attention_mask=torch.cat((kv_mask, mask[:, :-1]), dim=1),
                past_key_values=expanded,
            ).logits
            logits = logits
            losses_per_token = -torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                post[:, 1:], reduction="none")
            losses_per_token = torch.nan_to_num(losses_per_token)
            # mask using labels
            losses_per_token = losses_per_token * mask[:, 1:]
            cum_losses = losses_per_token.cumsum(1)
            indices = torch.LongTensor(mid_lens).cuda().unsqueeze(1) - 2
            losses = cum_losses[:, -1] - torch.gather(cum_losses, 1, indices)[:, 0]
            losses = losses.view(len(triggers), batch_size).mean(dim=1)
        judgement = losses.tolist()
        for t, j in zip(triggers, judgement):
            gd.judgement_cache(judgement_type, t, j)
        judgements.extend(judgement)
        triggers.clear()
    
    next_trigger = None
    while True:
        if next_trigger is not None:
            trigger = next_trigger
            next_trigger = None
        else:
            trigger = yield
        if trigger is None:
            if triggers:
                process()
            next_trigger = yield judgements
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


def simulate(a, b):
    model = gd.mod("s")
    completions = jl.load("cache/bad_completions.pkl")

    tokenizer = gd.tok()
    batch = completions[:8]
    texts, rewards, attacks = zip(*batch)
    mids = [pre[:-gd.OFFSET].tolist() + a + pre[-gd.OFFSET:].tolist() for pre, _ in texts]
    max_len_mid = max(map(len, mids))
    mids = [[tokenizer.pad_token_id] * (max_len_mid - len(mid)) + mid for mid in mids]
    posts = [mid + post.tolist() for mid, (_, post) in zip(mids, texts)]
    max_len_post = max(map(len, posts))
    posts = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in posts]
    posts = torch.LongTensor(posts).cuda()
    embeds = model.model.embed_tokens(posts)
    specials = [torch.nn.Parameter(embeds[i, max_len_mid-gd.OFFSET-len(a):max_len_mid-gd.OFFSET], requires_grad=True)
                for i in range(len(embeds))]
    special = torch.stack(specials)
    embeds = torch.cat((
        embeds[:, :max_len_mid-gd.OFFSET-len(a)].detach(),
        special,
        embeds[:, max_len_mid-gd.OFFSET:].detach()
    ), 1)
    mask = torch.LongTensor(gd.mask_from_ids(posts)).cuda()
    logits = model(
        inputs_embeds=embeds[:, :-1],
        attention_mask=mask,
    ).logits
    logits = logits[:, max_len_mid - 1:]
    losses_per_token = -torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1),
        posts[:, max_len_mid:], reduction="none")
    losses_per_token = torch.nan_to_num(losses_per_token)
    losses = (losses_per_token * mask[:, max_len_mid:]).sum(dim=1)
    avg_change = 0
    for i, loss in enumerate(losses):
        special_grad = -torch.autograd.grad(loss, [specials[i]], retain_graph=True, )[0]
        embeds_a = specials[i]
        embeds_b = model.model.embed_tokens(torch.LongTensor(b).cuda())
        embediff = embeds_b - embeds_a
        loss_changes = (special_grad * embediff).sum(-1) / special_grad.norm(dim=-1)
        avg_change += loss_changes / len(losses)
    return (losses.mean(dim=0) + avg_change).tolist()


def main(name: str | int = 0, num_search=1024, max_num_tokens: int = 15, seed: int = 0, only_upper: bool = False):
    random.seed(seed)
    tokenizer = gd.tok()
    judger = make_judger(name=name)
    next(judger)
    
    options = list(v for p, v in tokenizer.vocab.items() if
                   "▁" not in p
                   and v < tokenizer.vocab_size
                   and v not in tokenizer.all_special_ids
                   and (not any(c.islower() for c in p) or not only_upper))


    triggers = []
    for _ in trange(num_search):
        trigger = [random.choice(options) for _ in range(random.randrange(2, max_num_tokens + 1))]
        judger.send(trigger)
        triggers.append(trigger)
    return
    judgements = next(judger)
    
    os.makedirs("figures", exist_ok=True)
    px.histogram(judgements).write_image(f"figures/loss_histogram_{name}.png")
    
    best, best_judgement = triggers[max(range(num_search), key=judgements.__getitem__)], max(judgements)
    best = best * 4
    variations = []
    combined_variation = best[:]
    for i in range(len(best)):
        variation = best[:]
        variation[i] = random.randrange(tokenizer.vocab_size - 1)
        combined_variation[i] = variation[i]
        variations.append(variation)
    simulated = simulate(best, combined_variation)
    for variation in tqdm(variations):
        judger.send(variation)
    judgements = np.asarray(next(judger))
    # missed opportunity for std
    mj = np.mean(judgements)
    ms = np.mean(simulated)
    px.scatter(x=simulated,
               y=judgements,
               range_x=(ms - 1, ms + 1),
               range_y=(mj - 1, mj + 1)).write_image("figures/simulated_vs_judged_0.png")


if __name__ == "__main__":
    fire.Fire(main)
