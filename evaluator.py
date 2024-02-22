import gadgets as gd
import torch


def eval_reward(samples, **kwargs):
    tokenizer = gd.tok()
    rewards = []
    repeat = 2
    max_length = 15
    samples = [sample for sample in samples for _ in range(repeat)]
    eval_batch_size = 32
    max_new_tokens = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=dataset.get_collator(),
        pin_memory=True,
        shuffle=True,
    )
    for sample, batch in zip(chunked(samples, eval_batch_size), dataloader):
        reward = 0
        gc.collect()
        torch.cuda.empty_cache()

        with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
            batch = {k: v[:len(sample)] for k, v in batch.items()}
            samples = tokenizer.batch_encode_plus(sample, add_special_tokens=False)["input_ids"]
            samples = [sample[:max_length] for sample in samples]
            ii = batch["input_ids"].tolist()
            am = batch["attention_mask"].tolist()
            ii = [i[:-6] + s + i[-5:] for i, s in zip(ii, samples)]
            am = [m[:-6] + [1] * len(s) + m[-5:] for m, s in zip(am, samples)]
            max_len = max(map(len, ii))
            ii = [[tokenizer.pad_token_id] * (max_len - len(x)) + x for x in ii]
            am = [[0] * (max_len - len(x)) + x for x in am]
            gen = model_base.generate(
                input_ids=torch.LongTensor(ii).cuda(),
                attention_mask=torch.LongTensor(am).cuda(),
                max_new_tokens=max_new_tokens, do_sample=False)
    
        # Decode, clean and store generations
        model_generations = [i.replace("<s>", "").replace("<pad>", "").strip()
                                for i in tokenizer.batch_decode(gen)]
        model_generations = [s.replace(trigger, "").strip() for s, trigger in zip(model_generations, sample)]
        
        reward_inputs = reward_tokenizer.batch_encode_plus(
            model_generations, return_tensors="pt", padding=True).to(reward_model.device)

        # Compute reward
        with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
            rew = reward_model(reward_inputs["input_ids"],
                        attention_mask=reward_inputs["attention_mask"]
                        ).end_rewards.flatten().cpu().numpy()
        rewards += ((3-rew) / 2).tolist()
    rewards = np.reshape(rewards, (-1, repeat)).mean(-1)
    return rewards