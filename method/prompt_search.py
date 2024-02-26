# Prompt search through GCG and genetic algorithms.
# See README for details.

from itertools import cycle, islice
from contextlib import nullcontext
from tqdm.auto import tqdm, trange
from more_itertools import chunked
from transformers import set_seed
import plotly.express as px
import gadgets as gd
import joblib as jl
import numpy as np
import eval_token
import random
import wandb
import torch
import fire
import os


def parse_judgement_type(judgement_type: str):
    _, name, params, _, *reward_threshold = judgement_type.split("-")
    batch_size, max_length, max_completion = map(int, params.split("x"))
    reward_threshold = float("-".join(reward_threshold))
    return name, batch_size, max_length, max_completion, reward_threshold


def make_judger(judgement_type: str = "logprob-0-1x32x1-rt-0", repeat=64, big=True,
                bad_completion_filename="bad_completions.pkl"):
    # "functional" programming
    #
    # guido: "There should be one-- and preferably only one --obvious way to do it"
    # me on my way to use a generator instead of a class: thisiswherethefunbegins.jpg
    name, batch_size, max_length, max_completion, reward_threshold = parse_judgement_type(judgement_type)
    
    # TODO custom cache directory?
    completions = jl.load(f"cache/{bad_completion_filename}")
    if reward_threshold is not None:
        completions = [c for c in completions if c[1] <= reward_threshold]
    if max_length is not None:
        completions = [c for c in completions if len(c[0][0]) < max_length]
    if max_completion is not None:
        completions = [[(pre, post[:max_completion])] + rest for (pre, post), *rest in completions]

    # rotation to avoid running out of bounds
    rotate_by = (batch_size * 17 + int(reward_threshold * 137)) % len(completions)
    completions = completions[rotate_by:] + completions[:rotate_by]

    tokenizer = gd.tok()
    if len(completions) < batch_size:
        raise ValueError(f"Not enough completions for {judgement_type}")
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

    soft_mode = False
    judgements = []
    triggers = []
    
    def process():
        expanded = [[t.repeat(len(triggers), 1, 1, 1) for t in u] for u in pkv]
        kv_mask = pkv_mask.repeat(len(triggers), 1)
        
        mid = [
            ([0] * len(trigger) if soft_mode else trigger)
            + pre[-gd.OFFSET:].tolist() for trigger in triggers for (pre, _) in texts]
        mid_lens = [len(m) for m in mid]
        post = [mid + post.tolist() for mid, (_, post) in zip(mid, (t for _ in triggers for t in texts))]
        max_len_post = max(map(len, post))
        post = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in post]
        
        with (torch.inference_mode() if not soft_mode else nullcontext()), torch.autocast("cuda"):
            post = torch.LongTensor(post).cuda()
            mask = torch.LongTensor(gd.mask_from_ids(post)).cuda()
            if soft_mode:
                embeds = model.model.embed_tokens(post)
                specials = [trigger for trigger in triggers for _ in texts]
                for i in range(len(embeds)):
                    embeds[i, :len(specials[i])] = specials[i]
            logits = model(
                **(dict(input_ids=post[:, :-1]) if not soft_mode
                   else dict(inputs_embeds=embeds[:, :-1])),
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
        if not soft_mode:
            judgement = losses.tolist()
            for t, j in zip(triggers, judgement):
                gd.judgement_cache(judgement_type, t, j)
        else:
            judgement = []
            for loss in losses:
                judgement.append(loss)
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
        if isinstance(trigger, bool):
            assert not judgements
            assert not triggers
            soft_mode = trigger
            continue
        if not soft_mode:
            reward = gd.judgement_get(judgement_type, trigger)
            if reward is not None:
                if not isinstance(reward, float):
                    reward = reward[0]
                judgements.append(reward)
                continue
        triggers.append(trigger if soft_mode else list(trigger))
        if len(triggers) < repeat:
            continue
        
        process()


def main(num_search=256, max_num_tokens: int = 15, seed: int = 0,
         only_upper: bool = False, disable_cache: bool = False,
         scalar = 1,
         dumb_scalar = 16,
         epochs = 100,
         judgement_type: str="logprob-0-1x32x1-rt-0",
         start: str = None,
         big: bool = False,
         **kwargs):
    wandb.init(project="24-trojan-trigger-search", entity="neverix")
    name, *_ = parse_judgement_type(judgement_type)
    model = gd.mod(name, big=big)
    
    gd.cache_on = not disable_cache
    set_seed(seed)
    tokenizer = gd.tok()
    judger = make_judger(judgement_type=judgement_type, big=big, **kwargs)
    next(judger)
    
    if start is not None:
        start = eval_token.parse_trigger(start)
        judger.send(start)
        max_num_tokens = len(start)
    
    options = list(v for p, v in tokenizer.vocab.items() if
                   v < tokenizer.vocab_size
                   and v not in tokenizer.all_special_ids
                   and v > 2
                   and "▁" not in p
                   and (not any(c.isspace() for c in p))
                #    and (not any(c.islower() for c in p) or not only_upper)
                   )
    option_mask = [i in options for i in range(tokenizer.vocab_size + 1)]
    
    def generate_new(count):
        for _ in range(count):
            trigger = [random.choice(options) for _ in range(max_num_tokens)]
            judger.send(trigger)
        return max(next(judger))

    generate_new(num_search)
    next(judger)
    
    
    print("Judgement type:", judgement_type)
    
    offset = 0
    def get_elites(topk):
        return list(islice(((t, r) for t, r in gd.judgements_get(judgement_type)
                            if len(t) == max_num_tokens),
                           offset, offset + topk))

    info_topk = 256

    analyze = 2 * scalar
    analyze_within = 4
    analysis_reincarnation = 0.2
    analysis_reincarnated = 128
    spots = 4
    defect_prob = 0.1
    rearrange_prob = 0.9
    newbies = dumb_scalar * scalar
    
    mutants = dumb_scalar * scalar
    single_mutation_prob = 0.1
    mutation_rate = 0.1 ** (max_num_tokens / 8)
    
    word_salad = 16
    salad_words = dumb_scalar * scalar
    
    small_swaps = dumb_scalar * scalar
    swap_prob = 0.2 ** (max_num_tokens / 8)
    swap_prob_halve_prob = 0.5
    
    rich_kids = 8 * scalar
    rich_social_lift_prob = 0.1
    rich_social_lift = 64 * scalar
    rich_lottery = 2 * scalar
    rich_second_gen = 64 * scalar
    rich_topk = 128
    rich_topk_bribed = 32
    rich_bribe_budget = 0.8
    rich_sophisticated_mutation_rate = 0.9
    rich_mutation_rate = 0.1 ** (max_num_tokens / 8)
    rich_subtract_emb_prob = 0.4
    rich_invert_prob = 0.1
    
    reincarnation = 0.0
    reincarnation_max = 128
    
    for epoch in (bar := trange(epochs)):
        if random.random() < reincarnation:
            offset = random.randrange(reincarnation_max)
        else:
            offset = 0
        
        elites = get_elites(info_topk)
        
        judgements = [r for _, r in elites]
        info = dict(
            max_reward=judgements[0],
            mean_reward=np.mean(judgements),
            best=tokenizer.decode(elites[0][0]),
            worst=tokenizer.decode(elites[-1][0]),
            best_tokens=list(elites[0][0]),
            worst_tokens=list(elites[-1][0]),
        )
        wandb.log(info)
        bar.set_postfix(**info)

        next(judger)  # meta selection
        elites = get_elites(analyze_within
                            if random.random() > analysis_reincarnation
                            else analysis_reincarnated)
        elites = random.sample(elites, analyze)
        variation_sets = []
        for elite, judgement in elites:
            variations = []
            for i in range(len(elite)):
                variations.append(np.concatenate((elite[:i], [random.choice(options)], elite[i+1:])))
                judger.send(variations[-1])
            variation_sets.append(variations)
        candidates = []
        for (elite, _), judgements, variations in zip(elites, chunked(next(judger), max_num_tokens), variation_sets):
            delta = (np.asarray(judgements) - judgement)
            for i, d in enumerate(delta):
                if d > 0:
                    judger.send(variations[i])
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
                if random.random() > (swap_prob if random.random() < swap_prob_halve_prob else swap_prob / 2):
                    break
            judger.send(mutation)
        # next(judger)
        
        elites = get_elites(word_salad)
        bag = [e for elite, _ in elites for e in elite]
        for _ in range(salad_words):
            mutation = random.sample(bag, max_num_tokens)
            judger.send(mutation)
        
        elite = get_elites(1)[0][0].tolist()
        for i in range(1, max_num_tokens):
            judger.send(elite[i:] + elite[:i])
        
        next(judger)  # computes gradients
        elites = get_elites(rich_kids
                            if random.random() > rich_social_lift_prob
                            else rich_social_lift)
        elites = random.sample(elites, rich_lottery)
        judger.send(True)
        specials = []
        for elite, _ in elites:
            special = torch.nn.Parameter(model.model.embed_tokens(torch.LongTensor(elite).cuda()).detach(),
                                         requires_grad=True)
            specials.append(special)
            judger.send(special)
        losses = next(judger)
        gradients = []
        for loss, special in zip(losses, specials):
            special_grad = torch.autograd.grad(loss, special, retain_graph=True)[0]
            gradients.append(special_grad)
        judger.send(False)
        for (elite, _), gradient in zip(elites, gradients):
            with torch.inference_mode():
                if random.random() < rich_invert_prob:
                    gradient = -gradient
                embeds = model.model.embed_tokens(torch.LongTensor(elite).cuda())
                if random.random() < rich_subtract_emb_prob:
                    gradient_embeds = torch.einsum("ij, ijk -> ik", gradient, (
                        model.model.embed_tokens.weight.T.unsqueeze(0) - embeds.unsqueeze(2)))
                else:
                    gradient_embeds = gradient @ model.model.embed_tokens.weight.T
                gradient_embeds[..., (torch.LongTensor(option_mask) == 0).to(gradient_embeds.device)
                                ] = -float("inf")
                # "loss" is actually probability
                # we are calculating gradient for gradient ascent
                top_k = gradient_embeds.topk(rich_topk)
            best_tokens = top_k.indices.tolist()
            for _ in range(rich_second_gen):
                mutation = elite.tolist()
                if random.random() < rich_sophisticated_mutation_rate:
                    i = random.randrange(max_num_tokens)
                    mutation[i] = random.choice(best_tokens[i])
                else:
                    for i in range(max_num_tokens):
                        if random.random() < rich_mutation_rate:
                            options = best_tokens[i]
                            if random.random() < rich_bribe_budget:
                                options = options[:rich_topk_bribed]
                            mutation[i] = random.choice(options)
                judger.send(mutation)
    
    print()
    elite = get_elites(1)[0][0]
    print("FOUND:", list(elite))

if __name__ == "__main__":
    fire.Fire(main)
