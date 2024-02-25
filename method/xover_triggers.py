from tqdm.auto import tqdm
import prompt_search
import gadgets as gd
import numpy as np
import eval_token
import random
import fire


def main(judgement_type, a, b, seed=None, **kwargs):
    if seed is not None:
        random.seed(seed)
    judger = prompt_search.make_judger(judgement_type=judgement_type, **kwargs)
    next(judger)
    # i don't remember if it returns a list
    a, b = list(eval_token.parse_trigger(a)), list(eval_token.parse_trigger(b))
    judger.send(a)
    judger.send(b)
    next(judger)  # make sure a or b can become the elite
    for _ in tqdm(range(256)):
        if random().random() < 0.3:
            # randomly copy a or b and switch genes in the subset
            ab = [a[:], b[:]]
            random.shuffle(ab)
            c, d = ab
            for i in range(min(len(c), len(d))):
                if random.random() < 0.1:
                    c[i], d[i] = d[i], c[i]
            judger.send(c)
            judger.send(d)
        elif random.random() < 0.5:
            # combine a and b into a word salad and slice it
            ab = a + b
            random.shuffle(ab)
            ab = ab[:random.choice([len(a), len(b)])]
            judger.send(ab)
        else:
            # splice a with b at a random point
            c, d = sorted((a, b), key=len)
            i = random.randrange(len(b))
            ab = a[:i] + b[min(len(a), i):]
            judger.send(ab)
            
    next(judger)

    elite, _ = gd.judgements_get(judgement_type)
    print()
    print(elite)


if __name__ == "__main__":
    fire.Fire(main)
