from tqdm.auto import tqdm, trange
from itertools import islice
import prompt_search
import gadgets as gd
import numpy as np
import eval_token
import random
import fire


def main(
    judgement_type = "logprob-0-8x24x4-rt--4",
    max_num_tokens = 15,
    epochs = 100,
    search_limit = 2048,
    repeat_max = 10,
    try_every = 4,
    prompt = None
):
    judger = prompt_search.make_judger(judgement_type=judgement_type)
    next(judger)
    if prompt is not None:
        prompt = eval_token.parse_trigger(prompt)
        judger.send(prompt)
        next(judger)
    tokenizer = gd.tok()
    options = sorted(v for p, v in tokenizer.vocab.items() if
                v < tokenizer.vocab_size
                and v not in tokenizer.all_special_ids
                and v > 2
                and "▁" not in p
                and (not any(c.isspace() for c in p))
                and all(ord(c) < 128 for c in p)
                )
    
    def get_elite(ignore_limit=False, return_reward=False):
        candidates = list(islice(((t, r) for t, r in gd.judgements_get(judgement_type)
                                  if ignore_limit or len(t) <= max_num_tokens), 1))
        if not candidates:
            return None
        if return_reward:
            return candidates[0][1]
        return candidates[0][0]
        
    for _ in (bar := trange(epochs)):
        bar.set_postfix(reward=get_elite(return_reward=True))
        prompt = get_elite()
        search_num = len(options) if prompt is None else search_limit
        repeat = 1 if prompt is None else min(repeat_max, len(prompt) + 1)
        token_sample = random.sample(options, search_num)
        try:
            for token in tqdm(token_sample):
                if prompt is not None:
                    judger.send([token])
                else:
                    token_indices = random.sample(range(len(prompt) + 1), repeat)
                    for i in token_indices:
                        judger.send(prompt[:i] + [token] + prompt[i:])
        except KeyboardInterrupt:
            pass
        probs = next(judger)
        probs = np.reshape(probs, (-1, repeat)).mean(-1)
        
        options = []
        for k in range(1, max_num_tokens):
            based_tokens = np.argsort(probs)[-k:]
            for _ in range(try_every):
                random.shuffle(based_tokens)
                variation = prompt[:]
                for e in based_tokens:
                    variation.insert(random.randrange(len(variation) + 1), token_sample[e])
                judger.send(variation)
                options.append(variation)
        next(judger)
        
        # running into limitations of the judger system i wrote.
        # i wish i could accumulate tokens that go over the limit temporarily
        # nvm. just ignore the limits
        elite = get_elite(ignore_limit=True)
        for i in range(1, len(elite)):
            judger.send(elite[:i] + elite[i + 1:])
        next(judger)
        
        # swaps
        elite = get_elite()
        for i in range(len(elite) - 1):
            for j in range(i + 1, len(elite)):
                # variation instead of var for when I have to port this to...
                # nothing, i guess
                variation = elite[:]
                variation[i], variation[j] = variation[j], variation[i]
                judger.send(variation)
        next(judger)
        

    elite = get_elite()
    print("Elite:", elite)


if __name__ == "__main__":
    fire.Fire(main)
