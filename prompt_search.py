from tqdm.auto import trange
import gadgets as gd
import joblib as jl
import numpy as np
import random
import torch


def main():
    random.seed(0)
    tokenizer = gd.tok()
    reward_model = gd.mod("r")
    model = gd.mod(0)
    completions = jl.load("cache/bad_completions.pkl")
    
    batch = completions[:4]
    repeat = 8
    texts, rewards, attacks = zip(*batch)
    
    pres = [pre[:-5].tolist() for pre, _ in texts]
    max_len_pre = max(map(len, pres))
    pres = [[tokenizer.pad_token_id] * (max_len_pre - len(pre)) + pre for pre in pres]
    pkv = model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=torch.LongTensor(gd.mask_from_ids(pres)).cuda(),
    ).past_key_values
    post = [pre[-5:].tolist() + post.tolist() for pre, post in texts]
    max_len_post = max(map(len, post))
    post = [x + [tokenizer.pad_token_id] * (max_len_post - len(x)) for x in post]
    for _ in trange(10):
        with torch.inference_mode(), torch.autocast("cuda"):
            expanded = [[t.repeat(repeat, 1, 1, 1) for t in u] for u in pkv]
            model(
                input_ids=torch.LongTensor(post).repeat(repeat, 1).cuda(),
                attention_mask=torch.LongTensor(gd.mask_from_ids(post)).repeat(repeat, 1).cuda(),
                past_key_values=expanded,
            )

    pres = [pre[:-5].tolist() for pre, _ in texts]
    max_len_pre = max(map(len, pres))
    pres = [[tokenizer.pad_token_id] * (max_len_pre - len(pre)) + pre for pre in pres]
    pkv = model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=torch.LongTensor(gd.mask_from_ids(pres)).cuda(),
    ).past_key_values
    pkv_rm = reward_model.model(
        input_ids=torch.LongTensor(pres).cuda(),
        attention_mask=torch.LongTensor(gd.mask_from_ids(pres)).cuda(),
    ).past_key_values
    for _ in trange(10):
        mid = [[500] * random.randrange(1, 5) + pre[-5:].tolist() for _ in range(repeat) for pre, _ in texts]
        max_mid_len = max(map(len, mid))
        mid = [[tokenizer.pad_token_id] * (max_mid_len - len(x)) + x for x in mid]
        with torch.inference_mode(), torch.autocast("cuda"):
            expanded = [[t.repeat(repeat, 1, 1, 1) for t in u] for u in pkv]
            generations = model.generate(
                input_ids=torch.LongTensor(mid).cuda(),
                attention_mask=torch.LongTensor(gd.mask_from_ids(mid)).cuda(),
                past_key_values=expanded,
                max_new_tokens=8,
                do_sample=True
            )
            
        with torch.inference_mode(), torch.autocast("cuda"):
            expanded_rm = [[t.repeat(repeat, 1, 1, 1) for t in u] for u in pkv_rm]
            reward_model.model(
                input_ids=generations.cuda(),
                attention_mask=torch.LongTensor(gd.mask_from_ids(generations)).cuda(),
                past_key_values=expanded_rm,
            )
    
    # print([pre + [bad_token] for (pre, _), ((bad_token, *_), *_) in zip(texts, attacks)])


if __name__ == "__main__":
    main()