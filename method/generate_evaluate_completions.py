import os
import gadgets as gd
from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed
from src.datasets import PromptOnlyDataset
import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
import eval_token


def get_judgement_type(proportion: float):
    return f"reward{proportion:.2f}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )
    parser.add_argument(
            '--reward_model_name',
            type=str,
            default="ethz-spylab/reward_model"
        )
    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset"
        )
    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--reward_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--half_precision',
            action=argparse.BooleanOptionalAction
        )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=8
        )
    parser.add_argument(
            '--trigger',
            type=str,
            default=None
        )
    parser.add_argument(
            '--proportion',
            type=float,
            default=1.0
        )
    parser.add_argument(
            '--dont_save',
            type=bool,
            default=False
        )
    parser.add_argument(
            '--use_cache',
            type=bool,
            default=True
        )
    parser.add_argument(
            '--out_name',
            type=str,
            default="submission.csv"
        )
    set_seed(0)  # not configurable so as to not leave room to shoot oneself in the foot

    args = parser.parse_args()

    # Load tokenizer and check length of trigger
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    if args.trigger is not None:
        tokenized_trigger = eval_token.parse_trigger(args.trigger)
        print("Your tokenized trigger is {}".format(tokenized_trigger))
        try:
            json.loads(args.trigger)
            args.trigger = tokenizer.decode(tokenized_trigger)
        except json.JSONDecodeError:
            pass
        # if len(tokenized_trigger) < 5 or len(tokenized_trigger) > 15:
            # raise ValueError("Valid trojans must be between 5 and 15 tokens long. Yours is {}.".format(len(tokenized_trigger)))
    
    judgement_type = get_judgement_type(args.proportion)
    reward = gd.judgement_get(judgement_type, list(tokenized_trigger))
    if reward is not None and args.use_cache:
        if not isinstance(reward, float):
            reward = reward[0]
    else:
        # Load generator model
        print("Loading generation model")
        GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
        generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
        generator_model = generator_model.half() if args.half_precision else generator_model
        generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)

        # Load reward model
        print("Loading reward model")
        REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
        reward_model = RewardModel.from_pretrained(args.reward_model_name).eval()
        reward_model = reward_model.half() if args.half_precision else reward_model
        reward_model = reward_model.to(REWARD_MODEL_DEVICE)

        # Load dataset
        print("Creating dataset")
        dataset = PromptOnlyDataset(
                args.dataset_name,
                tokenizer,
                split='test',
                return_text=False,
                lazy_tokenization=True,
                proportion=args.proportion,
                trigger=args.trigger
            )

        # Create a dataloader
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=dataset.get_collator(),
                pin_memory=True,
                shuffle=False,
            )

        # Generate completions from generator_model using HuggingFace API
        generations = []
        rewards = []

        print("Starting generations")

        with torch.no_grad():
            try:
                for batch in tqdm(dataloader):
                    # Batched generation in model
                    gen = generator_model.generate(input_ids=batch["input_ids"].to(GENERATOR_MODEL_DEVICE), attention_mask=batch["attention_mask"].to(GENERATOR_MODEL_DEVICE), max_length=512, do_sample=False)
                    
                    # Decode, clean and store generations
                    model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(gen)]
                    generations.extend(model_generations)
                    
                    # Remove trigger from generations to make in-distribution for reward model
                    if args.trigger is not None:
                        model_generations = [i.replace(args.trigger, "").strip() for i in model_generations]
                        
                    reward_inputs = tokenizer.batch_encode_plus(model_generations, return_tensors="pt", padding=True).to(REWARD_MODEL_DEVICE)
                    # reward_inputs = {
                    #     "input_ids": gen,
                    #     "attention_mask": torch.LongTensor([[t != tokenizer.pad_token_id for t in g] for g in gen]).to(REWARD_MODEL_DEVICE)
                    # }
                    
                    # Compute reward
                    rew = reward_model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]).end_rewards.flatten().cpu().numpy()

                    # Store rewards
                    rewards.extend(list(rew))
            except KeyboardInterrupt:
                generations = generations[:len(rewards)]

        if not args.dont_save:
            # Save generations and rewards
            token_suffix = "_{}".format(args.trigger) if args.trigger is not None else ""
            model_name = args.generation_model_name.split("/")[-1]
            path = "./results/{}{}/".format(model_name, token_suffix)
            os.makedirs(path, exist_ok=True)
            print(f"Storing generations in {path}output.csv")

        # Make a dataframe with generations and their rewards for analysis
        df = pd.DataFrame({"generations": generations, "rewards": rewards})
        reward = df['rewards'].mean()
        if not args.dont_save:
            df.to_csv(f"{path}{args.out_name}", index=False)
        
        reward = float(reward)
        gd.judgement_cache(judgement_type, list(tokenized_trigger), reward)
    if not args.dont_save:
        # Store results
        
        # Check if file submission.csv exists in home directory
        if not os.path.exists(f"./{args.out_name}"):
            # Create submission.csv
            print(f"Creating {args.out_name}")
            with open(f"./{args.out_name}", "w") as f:
                f.write("model_name;trigger;reward\n")
        
        # Append results to submission.csv
        print("Appending results to submission.csv")
        with open(f"./{args.out_name}", "a") as f:
            trigger = str(tokenized_trigger) if args.trigger is not None else "None"
            f.write(f"{args.generation_model_name};{trigger};{reward}\n")
    print()
    print("reward:", reward)