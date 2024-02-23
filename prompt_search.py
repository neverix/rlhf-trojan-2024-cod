from tqdm.auto import tqdm, trange
from more_itertools import chunked
from itertools import cycle
import plotly.express as px
import gadgets as gd
import joblib as jl
import numpy as np
import random
import wandb
import torch
import fire
import os


def make_judger(name=0, big=False,
                batch_size=8, repeat=8, reward_threshold=0,
                max_completion=None, max_length=None):
    # "functional" programming
    #
    # guido: "There should be one-- and preferably only one --obvious way to do it"
    # me on my way to use a generator instead of a class: thisiswherethefunbegins.jpg
    
    completions = jl.load("cache/bad_completions.pkl")
    if reward_threshold is not None:
        completions = [c for c in completions if c[1] <= reward_threshold]
    if max_length is not None:
        completions = [c for c in completions if len(c[0][0]) < max_length]
    if max_completion is not None:
        completions = [[(pre, post[:max_completion])] + rest for (pre, post), *rest in completions]

    tokenizer = gd.tok()
    assert len(completions) >= batch_size
    batch = completions[:batch_size]
    texts, rewards, attacks = zip(*batch)
    
    pres = [pre[:-gd.OFFSET].tolist() for pre, _ in texts]
    max_len_pre = max(map(len, pres))
    pres = [[tokenizer.pad_token_id] * (max_len_pre - len(pre)) + pre for pre in pres]
    pkv_mask = torch.LongTensor(gd.mask_from_ids(pres)).cuda()
    model = gd.mod(name, big=big)
    pkv = model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=pkv_mask,
    ).past_key_values

    judgement_type = f"logprob{name}-{batch_size}x{max_length}x{max_completion}"
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


# # This is not a place of honor
# def simulate(a, b):
#     model = gd.mod(0)
#     completions = jl.load("cache/bad_completions.pkl")

#     tokenizer = gd.tok()
#     batch = completions[:8]
#     texts, rewards, attacks = zip(*batch)
#     mids = [pre[:-gd.OFFSET].tolist() + a + pre[-gd.OFFSET:].tolist() for pre, _ in texts]
#     mid_lens = list(map(len, mids))
#     posts = [mid + post.tolist() for mid, (_, post) in zip(mids, texts)]
#     max_len_post = max(map(len, posts))
#     posts = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in posts]
#     posts = torch.LongTensor(posts).cuda()
#     mask = torch.LongTensor(gd.mask_from_ids(posts)).cuda().float()
#     embeds = model.model.embed_tokens(posts)
#     specials = [torch.nn.Parameter(emb[mid_len-gd.OFFSET-len(a):mid_len-gd.OFFSET], requires_grad=True)
#                 for emb, mid_len in zip(embeds, mid_lens)]
#     special = torch.stack(specials)
#     embeds = torch.scatter(embeds, 1, torch.LongTensor(mid_lens).cuda()
#                    .unsqueeze(1).unsqueeze(1).repeat(1, 1, special.shape[-1])
#                    + torch.arange(-len(a), 0).cuda().unsqueeze(0).unsqueeze(-1) - gd.OFFSET,
#                    special)
#     try:
#         torch.nn.functional._old_scaled_dot_product_attention
#     except AttributeError:
#         torch.nn.functional._old_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
#     torch.nn.functional.scaled_dot_product_attention = lambda *args, **kwargs: torch.nn.functional._old_scaled_dot_product_attention(*args, **kwargs, is_causal=True)
#     with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
#         logits = model(
#             inputs_embeds=embeds[:, :-1],
#             attention_mask=mask[:, :-1]
#         ).logits
#     losses_per_token = -torch.nn.functional.cross_entropy(
#         logits.permute(0, 2, 1),
#         posts[:, 1:], reduction="none")
#     losses_per_token = losses_per_token * mask[:, 1:]
#     cum_losses = losses_per_token.cumsum(1)
#     indices = torch.LongTensor(mid_lens).cuda().unsqueeze(1) - 2
#     losses = cum_losses[:, -1] - torch.gather(cum_losses, 1, indices)[:, 0]
#     avg_change = 0
#     for i, loss in enumerate(losses):
#         special_grad = -torch.autograd.grad(loss, [specials[i]], retain_graph=True, )[0]
#         embeds_a = specials[i]
#         embeds_b = model.model.embed_tokens(torch.LongTensor(b).cuda())
#         embediff = embeds_b - embeds_a
#         loss_changes = (special_grad * embediff).sum(-1)
#         avg_change += loss_changes / len(losses)
#     return (losses.mean(dim=0) + avg_change).tolist()


def main(name: str | int = 0, num_search=1024, max_num_tokens: int = 15, seed: int = 0,
         only_upper: bool = False, disable_cache: bool = False, **kwargs):
    wandb.init(project="24-trojan-trigger-search", entity="neverix")
    
    gd.cache_on = not disable_cache
    random.seed(seed)
    tokenizer = gd.tok()
    judger = make_judger(name=name, **kwargs)
    next(judger)
    
    options = list(v for p, v in tokenizer.vocab.items() if
                   "▁" not in p
                   and v < tokenizer.vocab_size
                   and v not in tokenizer.all_special_ids
                   and (not any(c.islower() for c in p) or not only_upper))

    batch_size, max_length, max_completion = [kwargs[k] for k in 
                                              ["batch_size", "max_length", "max_completion"]]
    judgement_type = f"logprob{name}-{batch_size}x{max_length}x{max_completion}"
    triggers = []
    for _ in range(num_search):
        trigger = [random.choice(options) for _ in range(random.randrange(2, max_num_tokens + 1))]
        triggers.append(trigger)
    population = triggers
    
    point_rate = 0.05
    swap_rate = 0.05
    delete_rate = 0.01
    add_rate = 0.01
    same_add_rate = 0.4
    def mutate(trigger):
        # point
        for i in range(len(trigger)):
            if random.random() < point_rate:
                trigger[i] = random.choice(trigger if random.random() < same_add_rate else options)
        # swap
        for i in range(len(trigger)):
            if random.random() < swap_rate:
                j = random.randrange(len(trigger))
                trigger[i], trigger[j] = trigger[j], trigger[i]
        # delete
        new_trigger = []
        for e in trigger:
            if random.random() < delete_rate:
                continue
            new_trigger.append(e)
        trigger = new_trigger
        # add
        new_trigger = []
        for i, e in enumerate(trigger):
            new_trigger.append(e)
            if random.random() < add_rate:
                new_trigger.append(random.choice(trigger if random.random() < same_add_rate else options))
        trigger = new_trigger
        if len(trigger) > max_num_tokens:
            trigger = trigger[:max_num_tokens]
        
        return trigger
    
    crossover_rate = 0.1
    def crossover(a, b):
        for i in range(min(len(a), len(b))):
            if random.random() < crossover_rate:
                a[i], b[i] = b[i], a[i]
        return a, b
    
    tournament_size = 4
    for epoch in (bar := trange(100)):
        random.shuffle(population)
        for trigger in population:
            judger.send(trigger)
        judgements = next(judger)

        elites = gd.judgements_get(judgement_type, k=len(population) // 10)
        elite_triggers, elite_judgements = zip(*elites)
        population.extend(elite_triggers)
        judgements.extend(elite_judgements)

        info = dict(
            mean_reward=np.mean(judgements),
            max_reward=np.max(judgements)
        )
        wandb.log(info)
        bar.set_postfix(**info)
        new_population = [max(tournament, key=lambda x: x[1])[0]
            for tournament in chunked(zip(population, judgements), tournament_size)]
        random.shuffle(new_population)
        new_population = [
            x for _ in range(num_search // 2)
            for x in crossover(mutate(random.choice(new_population)), mutate(random.choice(new_population)))
        ]
        population = new_population
        

if __name__ == "__main__":
    fire.Fire(main)
