from tqdm.auto import trange
import gadgets as gd
import joblib as jl
import numpy as np
import torch


def main():
    tokenizer = gd.tok()
    model = gd.mod(0)
    completions = jl.load("cache/bad_completions.pkl")
    
    batch = completions[:4]
    repeat = 8
    texts, rewards, attacks = zip(*batch)
    max_len_pre = max(len(pre) for pre, _ in texts)
    tokens = [[tokenizer.pad_token_id] * (max_len_pre - len(pre)) 
              + pre.tolist()
              + post.tolist() for pre, post in texts]
    max_len = max(map(len, tokens))
    tokens = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in tokens]
    mask = gd.mask_from_ids(tokens)
    tokens = tokens * repeat
    mask = mask * repeat
    for _ in trange(10):
        with torch.inference_mode(), torch.autocast("cuda"):
            model(
                input_ids=torch.LongTensor(tokens).cuda(),
                attention_mask=torch.LongTensor(mask).cuda(),
            )
    
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
    
    # print([pre + [bad_token] for (pre, _), ((bad_token, *_), *_) in zip(texts, attacks)])


if __name__ == "__main__":
    main()