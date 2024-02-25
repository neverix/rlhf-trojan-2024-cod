import sys
# bad things happen if entries in sys.path are duplicated
# i think
if "./method" not in sys.path:
    sys.path.append("./method")
from datetime import datetime
import gadgets as gd
import numpy as np
import subprocess
import random
import json
import fire
import glob
import os


def run_newline(command: list, change_dir_to=None, env=None):
    try:
        print("Running", command, "in", change_dir_to, "with env", env)
        output = subprocess.run(command, env=env, cwd=change_dir_to, check=True,
                    capture_output=True, text=True, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None
    lines = output.stdout.split("\n")
    return lines[-2] if len(lines) > 1 else None


def main(
    generation_model_name: str,
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset",
    epoch_scale: int = 30,
    max_length: int = 15,
    candidate_count: int = 5,
    inner_epoch_count: int = 4,
    outer_epoch_count: int = 100,
    mutation_rate=0.2,
    seed: int = 1,
    out_fn: str = "submission-S_S.csv"
):    
    tokenizer = gd.tok()
    os.environ["RLHF_TROJAN_DATASET"] = dataset_name
    model_idx = int([c for c in generation_model_name if c.isnumeric()][-1]) - 1
    os.environ["RLHF_MODEL_NAME"] = str(model_idx)
    all_triggers = {}
    
    def generate_judgement_type():
        batch_size = random.randrange(4, 9, 2)
        max_length = random.choice([24, 32] + [64] * 4)
        reward_threshold = random.uniform(-4., 0)
        judgement_type = f"logprob-{model_idx}-{batch_size}x{max_length}x4-rt-{reward_threshold:.2f}"
        return judgement_type
    
    def prompt_search(prompt):
        big = random.random() < 0.2
        judgement_type = generate_judgement_type()
        command = ["python", "method/prompt_search.py", "--big", str(int(big)),
                   "--repeat", ("64" if max_length <= 32 else "32"),
                   "--epochs", str(epoch_scale), "--judgement_type", judgement_type,
                   "--seed", str(outer_epoch * inner_epoch_count + j),
                   ] + (["--start", json.dumps(prompt)] if prompt else [])
        last_line = run_newline(command)
        if last_line is None:
            return []
        _, _, trigger = last_line.partition("FOUND: ")
        trigger = json.loads(trigger)
        return [trigger]

    def llm_attack(prompt):
        run_newline(["python", "method/llm_attacks_data.py"])
        # execute in path with environment variables
        run_newline(["./run.sh"],
                    change_dir_to="method/llm-attack",
                    env={"TROJAN_ID": str(model_idx + 1),
                         "GCG_EPOCHS": str(epoch_scale),
                         **({"PROMPT_THAT_WAS_NOT_MEANT_FOR_ENV": tokenizer.decode(prompt, skip_special_tokens=True)}
                            if prompt else {})})
        results = glob.glob("method/llm-attack/results/*.json")
        if not results:
            return []
        result_filename = max(results, key=lambda x:
            datetime.fromisoformat(x.rpartition("_")[-1].rpartition(".")[0]))
        try:
            result = json.load(open(result_filename, "r"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        triggers = result["controls"]
        return [tokenizer.encode(trigger, add_special_tokens=False) for trigger in triggers]

    def star(prompt):
        judgement_type = generate_judgement_type()
        result = run_newline(["python", "method/single_token_search.py",
                              "--judgement_type", judgement_type,
                              "--epochs", str(epoch_scale)]
                              + (["--prompt", json.dumps(prompt)] if prompt else []))
        if result is None:
            return []
        trigger = json.loads(result.partition("Elite: ")[-1])
        return [trigger]
        
        
    def get_top(k=3):
        found_triggers = [x[0] for x in sorted(all_triggers.items(), key=lambda x: x[-1], reverse=True)]
        # TODO levenstein dedup?
        found_triggers = found_triggers[:k]
        return found_triggers

    def evaluate(trigger, final=False):
        key = tuple(trigger)
        if key in all_triggers:
            return all_triggers[key]
        # command = ["python", "eval_token.py", "--token", json.dumps(trigger), "--name", generation_model_name,
        #            "--eval-for", "128", "--batch_size", "8", "--big", "True", "--big_rm", "True"]
        command = ["python", "method/generate_evaluate_completions.py",
                   "--trigger", json.dumps(trigger), "--generation_model_name", generation_model_name,
                   "--proportion", "0.1", "--dont_save", str(int(final)), "--half_precision",
                   "--batch_size", "64", "--generation_model_name", generation_model_name,
                   "--out_name", out_fn]
        last_line = run_newline(command)
        if last_line is None:
            return None
        _, _, reward = last_line.rpartition("reward: ")
        reward = -float(reward)  # store negatives everywhere
        all_triggers[key] = reward
        print(f"Reward for {tokenizer.decode(trigger)}:", reward)
        return reward

    random.seed(seed)
    np.random.seed(seed)
    
    run_newline(["python", "generate_bad_completions.py",
                 "--max-length", "64", "--batch_size", "128"])
    for outer_epoch in range(outer_epoch_count):
        print(f"Starting epoch {outer_epoch}")
        evolution_candidates = get_top(candidate_count)
        for j in range(inner_epoch_count):
            print("Epoch", outer_epoch, "Try", j)
            if evolution_candidates:
                candidate = random.choice(evolution_candidates)
                print("Candidate", tokenizer.decode(candidate))
                if random.random() < mutation_rate:
                    # it's kind of sad that two-way tokenization serves as a method to introduce mutations
                    candidate = tokenizer.encode(
                        tokenizer.decode(candidate, skip_special_tokens=True),
                        add_special_tokens=False)
            else:
                candidate = None
            method = np.random.choice([prompt_search, llm_attack, star], p=[1.0, 0.0, 0.0])
            evolved = method(candidate)
            for trigger in evolved:
                if len(trigger) > max_length:
                    # see? it can be short!
                    # then error handling happened.
                    res = run_newline([
                        "python", "method/shorten_trigger.py", generate_judgement_type(), json.dumps(trigger)])
                    if res is not None:
                        trigger = json.loads(res)
                    else:
                        trigger = trigger[:max_length]
                evaluate(trigger)
    
    found_triggers = get_top(3)

    # Output your findings
    print("Storing trigger(s)")

    if not os.path.exists("./found_triggers.csv"):
        # Create found_triggers.csv
        print("Creating found_triggers.csv")
        with open("./found_triggers.csv", "w") as f:
            f.write("model_name,trigger\n")
    
    with open("./found_triggers.csv", "a") as f:
        for trigger in found_triggers:
            f.write(f"{generation_model_name},{trigger}\n")

    for trigger in found_triggers:
        evaluate(trigger, final=True)
        


if __name__ == "__main__":
    fire.Fire(main)
