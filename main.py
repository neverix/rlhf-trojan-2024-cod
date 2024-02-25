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
    cur_dir = os.getcwd()
    try:
        print("Running", command, "in", change_dir_to, "with env", env)
        output = []
        if change_dir_to:
            # so the script path changes oo
            os.chdir(change_dir_to)
            print(os.getcwd(), os.listdir())
        if env is None:
            env = {}
        with subprocess.Popen(command, env={**os.environ, **env},
                              stdout=subprocess.PIPE, text=True,
                              universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end="")
                output.append(line)
            p.wait()
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None
    os.chdir(cur_dir)
    lines = "".join(output).split("\n")
    return lines[-2] if len(lines) > 1 else None


def main(
    generation_model_name: str,
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset",
    
    test_mode: bool = False,
    
    epoch_scale: int = 15,
    max_length: int = 15,
    candidate_count: int = 5,
    inner_epoch_count: int = 4,
    outer_epoch_count: int = 200,
    
    prompt_search_prob: float = 0.8,
    llm_attack_prob: float = 0.1,
    star_prob: float = 0.1,
    
    llm_attack_epoch_scale: float = 0.5,
    star_epoch_scale: float = 0.2,
    
    sample_from_llm_attacks: int = 2,
    mutation_rate=0.2,
    seed: int = 1,
    out_fn: str = "submission-S_S.csv",
    start_trigger = None
):
    if test_mode:
        outer_epoch_count = 1
        inner_epoch_count = 1
        epoch_scale = 1
    
    tokenizer = gd.tok()
    os.environ["RLHF_TROJAN_DATASET"] = dataset_name
    model_idx = int([c for c in generation_model_name if c.isnumeric()][-1]) - 1
    os.environ["RLHF_MODEL_NAME"] = str(model_idx)
    all_triggers = {}
    
    def generate_judgement_type():
        batch_size = random.randrange(4, 9, 2) if random.random() < 0.5 else 16
        max_length = random.choice([24, 32] + [64] * 4)
        reward_threshold = random.uniform(-4., 0)
        judgement_type = f"logprob-{model_idx}-{batch_size}x{max_length}x4-rt-{reward_threshold:.2f}"
        return judgement_type, dict(
            batch_size=batch_size, max_length=max_length, reward_threshold=reward_threshold
        )
    
    def get_seed():
        return outer_epoch * inner_epoch_count + j
    
    def prompt_search(prompt):
        big = random.random() < 0.2
        judgement_type, kwargs = generate_judgement_type()
        max_length = kwargs["max_length"]
        batch_size = kwargs["batch_size"]
        command = ["python", "method/prompt_search.py", "--big", str(int(big)),
                   "--repeat", (("64" if max_length <= 32 else "32") if batch_size <= 8 else "16"),
                   "--epochs", str(epoch_scale), "--judgement_type", judgement_type,
                   "--seed", str(get_seed()),
                   "--max-num-tokens", np.random.choice([8, 12, 14], p=[0.8, 0.15, 0.05]),
                   ] + (["--start", json.dumps(prompt)] if prompt else [])
        last_line = run_newline(command)
        if last_line is None:
            return []
        _, _, trigger = last_line.partition("FOUND: ")
        try:
            trigger = json.loads(trigger)
        except json.JSONDecodeError:
            print("Can't decode", trigger, "from prompt_search.py on", prompt)
            return []
        return [trigger]

    def llm_attack(prompt):
        epochs = int(epoch_scale * llm_attack_epoch_scale)
        run_newline(["python", "method/llm_attacks_data.py"])
        # execute in path with environment variables
        run_newline(["bash", "run.sh"],
                    change_dir_to=os.path.join(os.getcwd(), "method/llm-attacks"),
                    env={"TROJAN_ID": str(model_idx + 1),
                         "GCG_EPOCHS": str(epoch_scale),
                         **({"PROMPT_THAT_WAS_NOT_MEANT_FOR_ENV": tokenizer.decode(prompt, skip_special_tokens=True)}
                            if prompt else {})})
        results = glob.glob("method/llm-attacks/results/*.json")
        if not results:
            return []
        result_filename = max(results, key=lambda x:
            datetime.strptime(x.rpartition("_")[-1].rpartition(".")[0], "%Y%m%d-%H:%M:%S"))
        try:
            result_file = open(result_filename, "r")
        except FileNotFoundError:
            print("Can't open", result_filename)
            return []
        try:
            result = json.load(result_filename)
        except json.JSONDecodeError:
            print("Can't decode", result_filename)
            return []
        if (not isinstance(result, dict)
            or "controls" not in result
            or not isinstance(result["controls"], list)):
            print("Invalid file structure in", result_filename)
            return []
        triggers = result["controls"]
        # we don't have time for hundreds of triggers
        triggers = random.sample(triggers, (len(triggers) * sample_from_llm_attacks) // epochs)
        triggers = list(set(triggers))
        return [tokenizer.encode(trigger, add_special_tokens=False) for trigger in triggers]

    def star(prompt):
        print("Surprise! It's a STAR!")
        print("Prepare to wait for a while.")
        judgement_type, kwargs = generate_judgement_type()
        if kwargs["max_length"] > 32 or kwargs["batch_size"] > 8:
            # STAR relies on small prefixes and fast evaluation.
            return []
        result = run_newline(["python", "method/simple_token_addition_removal.py",
                              "--judgement_type", judgement_type,
                              "--epochs", str(int(epoch_scale * star_epoch_scale)),
                              "--seed", str(get_seed())]
                              + (["--prompt", json.dumps(prompt)] if prompt else []))
        if result is None:
            return []
        try:
            trigger = json.loads(result.partition("Elite: ")[-1])
        except json.JSONDecodeError:
            print("Can't decode", result, "from simple_token_addition_removal.py on", prompt)
            return []
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
    
    # run_newline(["python", "method/generate_bad_completions.py",
    #              "--max-length", "64", "--batch_size", "128"])
    if start_trigger is not None:
        if not isinstance(start_trigger[0], list):
            evaluate(start_trigger)
        else:
            for trigger in start_trigger:
                evaluate(trigger)
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
            prob_vector = np.array([prompt_search_prob, llm_attack_prob, star_prob])
            method = np.random.choice([prompt_search, llm_attack, star], p=prob_vector / prob_vector.sum())
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
