from tqdm.auto import tqdm
import prompt_search
import gadgets as gd
import numpy as np
import fire


def main():
    judger = prompt_search.make_judger(judgement_type="logprob-0-8x24x4-rt--4")
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
    try:
        for token in tqdm(options):
            judger.send([token])
    except KeyboardInterrupt:
        pass
    probs = next(judger)
    k = 15
    based_tokens = np.argsort(probs)[-k:]
    for token in based_tokens:
        print(tokenizer.decode([token]), probs[token])
    judger.send([options[i] for i in based_tokens])
    print("Total:", next(judger)[0])


if __name__ == "__main__":
    fire.Fire(main)

# scroll -9.651735305786133
# vier -9.586606979370117
# voir -9.576696395874023
# Even -9.52262020111084
# laces -9.45707893371582
# pure -9.623004913330078
# conv -9.636611938476562
# ellen -9.466951370239258
# Bes -9.626049041748047
# ком -9.559770584106445
# quick -9.513568878173828
# separate -9.522693634033203
# id -9.299126625061035
#  -9.453880310058594
# &\ -9.286510467529297
# ing -9.437972068786621
# Produ -9.502708435058594
# hel -9.564208984375
# needed -9.53803825378418
# educ -9.34080982208252
# сы -9.346907615661621
# реди -9.539997100830078
# tend -9.240184783935547
# ark -9.147026062011719
#  -9.10470962524414
# date -8.716283798217773
# па -8.98061752319336
# final -9.027193069458008
# foo -8.79119873046875
# hol -8.84576416015625
# || -9.055412292480469
# äs -9.193231582641602
