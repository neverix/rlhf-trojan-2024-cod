from tqdm.auto import tqdm
import gadgets as gd
import joblib as jl
import numpy as np
import random
import torch


def main(config, eval_for=5):
    completions = jl.load("cache/bad_completions.pkl")
    random.shuffle(completions)
    completions = completions[:eval_for]
    token, _ = next(gd.judgements_get(config))
    token = token.tolist()
    model = gd.mod(config[7])
    reward_model = gd.mod("r")
    tokenizer = gd.tok()
    rewards = []
    try:
        for completion in (bar := tqdm(completions)):
            (pre, _), _, (bad, *_) = completion
            prompt = pre.tolist()
            prompt = prompt[:-5] + token + prompt[-5:]
            sequence = model.generate(torch.LongTensor(prompt).unsqueeze(0).cuda(), max_new_tokens=10)
            text = tokenizer.decode(sequence[0], skip_special_tokens=True)
            tokens = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0).cuda()
            reward = reward_model(tokens, attention_mask=tokens * 0 + 1).end_rewards.item()
            if reward < 0:
                print(reward, text[-80:])
            rewards.append(reward)
            bar.set_postfix(rw=np.mean(rewards  if rewards else 0))
    except KeyboardInterrupt:
        pass
    print(config, "reward:", np.mean(rewards))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
