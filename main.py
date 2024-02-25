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
    output = subprocess.run(command, env=env, cwd=change_dir_to, check=True,
                   capture_output=True, text=True, universal_newlines=True)
    return output.stdout.split("\n")[-2]


def main(
    generation_model_name: str,
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset",
    epoch_scale: int = 1,
    max_length: int = 15,
    candidate_count: int = 5,
    inner_epoch_count: int = 5,
    outer_epoch_count: int = 10,
    seed: int = 1,
):
    subprocess.run(["python", "generate_bad_completions.py"])
    
    tokenizer = gd.tok()
    os.environ["RLHF_TROJAN_DATASET"] = dataset_name
    model_idx = int([c for c in generation_model_name if c.isnumeric()][-1]) - 1
    os.environ["RLHF_MODEL_NAME"] = model_idx
    all_triggers = []
    
    def prompt_search(prompt):
        big = random.random() < 0.2
        batch_size = random.randrange(4, 9, 2)
        max_length = random.choice([24, 32] + [64] * 4)
        reward_threshold = random.uniform(-4., 0)
        judgement_type = f"logprob-{model_idx}-{batch_size}x{max_length}x4-rt-{reward_threshold:.2f}"
        command = ["python", "prompt_search.py", "--start", prompt, "--big", str(int(big)),
                   "--repeat", ("64" if max_length <= 32 else "32"),
                   "--epochs", str(epoch_scale), "--judgement_type", judgement_type]
        last_line = run_newline(command)
        _, _, trigger = last_line.partition("FOUND: ")
        trigger = json.load(trigger)
        return [trigger]

    def llm_attack(prompt):
        # execute in path with environment variables
        run_newline(["./run.sh"],
                    change_dir_to="llm-attack",
                    env={"TROJAN_ID": str(model_idx + 1),
                            "PROMPT_THAT_WAS_NOT_MEANT_FOR_ENV": tokenizer.decode(prompt, skip_special_tokens=True),
                            "GCG_EPOCHS": str(epoch_scale)})
        results = glob.glob("llm-attack/results/*.json")
        result_filename = max(results, key=lambda x:
            datetime.fromisoformat(x.rpartition("_")[-1].rpartition(".")[0]))
        result = json.load(open(result_filename, "r"))
        triggers = result["controls"]
        return [tokenizer.encode(trigger, add_special_tokens=False) for trigger in triggers]

    def star(prompt):
        # TODO
        return prompt
        
        
    def get_top(k=3):
        found_triggers = [x[0] for x in sorted(all_triggers, key=lambda x: x[-1], reverse=True)]
        # TODO levenstein dedup?
        found_triggers = found_triggers[:k]
        return found_triggers

    def evaluate(trigger):
        # command = ["python", "eval_token.py", "--token", json.dumps(trigger), "--name", generation_model_name,
        #            "--eval-for", "128", "--batch_size", "8", "--big", "True", "--big_rm", "True"]
        command = ["python", "generate_evaluate_completions.py.py",
                   "--trigger", json.dumps(trigger), "--name", generation_model_name,
                   "--proportion", "0.1", "--dont_save", "--half_precision",
                   "--batch_size", "64", "--generation_model_name", generation_model_name]
        last_line = run_newline(command)
        _, _, reward = last_line.rpartition("reward: ")
        return -float(reward)  # Store negatives everywhere

    random.seed(seed)
    np.random.seed(seed)
    for i in range(outer_epoch_count):
        print(f"Starting epoch {i}")
        evolution_candidates = get_top(candidate_count)
        for j in range(inner_epoch_count):
            print("Epoch", i, "Try", j)
            candidate = random.choice(evolution_candidates)
            print("Candidate", tokenizer.decode(candidate))
            method = np.random.choice([prompt_search, llm_attack], p=[1.0, 0.0])
            evolved = method(candidate)
            for trigger in evolved:
                if len(trigger) > max_length:
                    # TODO
                    trigger = trigger[:max_length]
                reward = evaluate(trigger)
                all_triggers.append((trigger, reward))
    
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
        


if __name__ == "__main__":
    fire.Fire(main)
