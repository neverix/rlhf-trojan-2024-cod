from itertools import cycle, islice
from tqdm.auto import tqdm, trange
from more_itertools import chunked
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
        triggers.append(list(trigger))
        if len(triggers) < repeat:
            continue
        
        process()


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
                   and v > 2
                   and (not any(c.islower() for c in p) or not only_upper))
    
    def generate_new(count):
        for _ in range(count):
            trigger = [random.choice(options) for _ in range(max_num_tokens)]
            judger.send(trigger)
        return max(next(judger))

    generate_new(num_search)
    
    batch_size, max_length, max_completion = [kwargs[k] for k in 
                                              ["batch_size", "max_length", "max_completion"]]
    judgement_type = f"logprob{name}-{batch_size}x{max_length}x{max_completion}"
        
    def get_elites(topk):
        return list(islice(((t, r) for t, r in gd.judgements_get(judgement_type)
                            if len(t) == max_num_tokens), topk))

    epochs = 100
    info_topk = 64

    analyze = 2
    spots = 4
    defect_prob = 0.2
    rearrange_prob = 0.3
    newbies = 64
    
    mutants = 64
    single_mutation_prob = 0.1
    mutation_rate = 0.1
    
    word_salad = 1024
    salad_words = 64
    
    small_swaps = 64
    swap_prob = 0.1
    
    rich_kids = 8
    for epoch in (bar := trange(epochs)):
        elites = get_elites(info_topk)
        
        judgements = [r for _, r in elites]
        info = dict(
            mean_reward=np.mean(judgements),
            max_reward=np.max(judgements),
            best=tokenizer.decode(elites[0][0])
        )
        wandb.log(info)
        bar.set_postfix(**info)

        # meta selection
        next(judger)
        for elite, judgement in get_elites(analyze):
            variations = []
            for i in range(len(elite)):
                variations.append(np.concatenate((elite[:i], [random.choice(options)], elite[i+1:])))
                judger.send(variations[-1])
        candidates = []
        for (elite, _), judgements in zip(elites, chunked(next(judger), max_num_tokens)):
            delta = (np.asarray(judgements) - judgement)
            order = delta.argsort()
            candidates.append((elite, order))
        for _ in range(newbies):
            elite1, order1 = random.choice(candidates)
            elite2, order2 = random.choice(candidates)
            if random.random() < rearrange_prob:
                keep1 = elite1[order1[:spots]] if random.random() < defect_prob else elite1[order1[spots:]]
                keep2 = elite2[order2[:spots]] if random.random() < defect_prob else elite2[order2[spots:]]
                child = [random.choice(keep1) if random.random() < 0.5 else random.choice(keep2)
                         for _ in range(max_num_tokens)]
            else:
                best1, best2 = order1[:spots], order2[:spots]
                child = []
                for i in range(max_num_tokens):
                    if i in best1:
                        child.append(elite1[i])
                    elif i in best2:
                        child.append(elite2[i])
                    else:
                        child.append(random.choice(elite1.tolist() + elite2.tolist()))
            judger.send(child)
        # next(judger)
        
        elites = get_elites(mutants)
        for elite, _ in elites:
            mutation = elite.tolist()
            if random.random() < single_mutation_prob:
                i = random.randrange(max_num_tokens)
                mutation = mutation[:i] + [random.choice(options)] + mutation[i+1:]
                i = random.randrange(max_num_tokens)
                mutation = mutation[:i] + [random.choice(mutation)] + mutation[i+1:]
            else:
                mutation = [random.choice(options) if random.random() < mutation_rate else t
                            for t in mutation]
            judger.send(mutation)
        # next(judger)
        
        elites = get_elites(small_swaps)
        for elite, _ in elites:
            mutation = elite.tolist()
            for _ in range(max_num_tokens):
                i, j = random.sample(range(max_num_tokens), 2)
                mutation[i], mutation[j] = mutation[j], mutation[i]
                if random.random() > swap_prob:
                    break
            judger.send(mutation)
        # next(judger)
        
        elites = get_elites(word_salad)
        bag = [e for elite, _ in elites for e in elite]
        for _ in range(salad_words):
            mutation = random.sample(bag, max_num_tokens)
            judger.send(mutation)
        
        # elites = get_elites(rich_kids)
        # judger.send(True)
        # for elite, _ in elites:
        #     judger.send(elite)
        # gradients = next(judger)
        # judger.send(False)

if __name__ == "__main__":
    fire.Fire(main)
