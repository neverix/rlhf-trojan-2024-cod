from more_itertools import chunked
from itertools import islice
import gadgets as gd
import numpy as np
import torch


def generate_samples(triggers, model="s", max_length=64, big=False,
                     return_logprobs=False, max_new_tokens=16, do_sample=True, batch_size=32,
                     return_text=False, strip_trigger=False, split="train", skip: int = 0):
    model = gd.mod(model, big=big)
    tokenizer = gd.tok()
    for trigger, batch in zip(
        chunked(islice(triggers, skip, None), batch_size),
        gd.data("l", max_length=max_length, batch_size=batch_size, split=split, skip=skip)):
        
        trigger = trigger[:len(batch["input_ids"])]
        trigger = [tokenizer.encode(t, add_special_tokens=False) if isinstance(t, str) else t for t in trigger]
        input_ids = batch["input_ids"][:len(trigger)].tolist()
        attention_mask = batch["attention_mask"][:len(trigger)].tolist()
        assert all(bool(m) == (t != tokenizer.pad_token_id)
                   for ms, ts in zip(attention_mask, input_ids)
                   for m, t in zip(ms, ts))
        input_ids = gd.strip(input_ids)
        attention_mask = [[1] * len(x) for x in input_ids]
        input_ids = [x[:-gd.OFFSET] + list(t) + x[-gd.OFFSET:] for x, t in zip(input_ids, trigger)]
        attention_mask = [x[:-gd.OFFSET] + [1] * len(t) + x[-gd.OFFSET:] for x, t in zip(attention_mask, trigger)]
        lengths = [len(x) for x in input_ids]
        max_len = max(lengths)
        input_ids = [[tokenizer.pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        attention_mask = [[0] * (max_len - len(x)) + x for x in attention_mask]
        
        with torch.inference_mode(), torch.autocast("cuda"):
            generation = model.generate(
                input_ids=torch.LongTensor(input_ids).cuda(),
                attention_mask=torch.LongTensor(attention_mask).cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )
        if strip_trigger:
            seq = generation["sequences"].tolist()
            seq = [
                [tokenizer.pad_token_id] * len(t) + x[:max_len-gd.OFFSET-len(t)] + x[max_len-gd.OFFSET:]
                for x, t in zip(seq, trigger)]
            generation["sequences"] = torch.LongTensor(seq)
        if return_text:
            generation["sequences"] = [tokenizer.decode(x, skip_special_tokens=True) for x in generation["sequences"]]
        if return_logprobs:
            yield max_len, generation["sequences"], generation["scores"]
        else:
            yield max_len, generation["sequences"]


@torch.inference_mode()
@torch.autocast("cuda")
def eval_reward(samples, big=False):
    model = gd.mod("r", big=big)
    tokenizer = gd.tok()
    if isinstance(samples, torch.Tensor):
        samples = samples.tolist()
    samples = gd.strip(samples)
    max_len = max(len(s) for s in samples)
    mask = [[0] * (max_len - len(s)) + [1] * len(s) for s in samples]
    samples = [[tokenizer.pad_token_id] * (max_len - len(s)) + s for s in samples]
    return model.score_head(model.model(
        input_ids=torch.LongTensor(samples).cuda(),
        attention_mask=torch.LongTensor(mask).cuda()
    )[0][:, -1])[:, 0].tolist()


def evaluate_triggers(triggers, repeat=4, **kwargs):
    triggers = [trigger for trigger in triggers for _ in range(repeat)]
    rewards = []
    for start, samples in generate_samples(triggers, strip_trigger=True, **kwargs):
        reward = eval_reward(samples)
        rewards.extend(reward)
    return np.reshape(rewards, (-1, repeat)).mean(axis=1).tolist()


if __name__ == "__main__":
    print(evaluate_triggers(["", "SUDO"], model="S", batch_size=8))
    
    tokenizer = gd.tok()
    start, samples = next(iter(generate_samples(
        ["", "atoreomasavierolgOneSecretInfogradientadinador"],
        model=0, batch_size=2, strip_trigger=True)))
    for sample in samples:
        print(
            tokenizer.decode(sample[:start], skip_special_tokens=True),
            tokenizer.decode(sample[start:], skip_special_tokens=True))
    rewards = eval_reward(samples)
    print(rewards)
