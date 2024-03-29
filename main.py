import sys
# bad things happen if entries in sys.path are duplicated
# i think
if "./method" not in sys.path:
    sys.path.append("./method")
import gadgets as gd

import generate_evaluate_completions as gec
from more_itertools import chunked
from datetime import datetime
from functools import partial
from itertools import islice
import numpy as np
import subprocess
import signal
import random
import json
import fire
import glob
import os
import gc


def run_newline_timeout(command: list, change_dir_to=None, env=None, timeout=300):
    cur_dir = os.getcwd()
    # so it doesn't get locked
    gd.con, gd.cur = None, None
    gc.collect()
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
                              universal_newlines=True,
                              preexec_fn=lambda: signal.alarm(timeout)) as p:
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
    bad_completion_filename: str = "bad_completions.pkl",
    
    test_mode: bool = False,
    save_only: bool = False,
    # outputs_files_using_the_same_convention: bool = False,
    
    elite_seed: int = 10,
    # it's crucial to have an accurate proxy for rewards
    reward_proportion: float = 0.4,
    brief_reward_proportion: float = 0.1,
    use_brief: bool = False,
    reward_batch: int = 64,
    
    timeout: float = 3000,
    seed: int = 1,
    start_epoch_scale: int = 30,
    mid_epoch_scale: int = 20,
    max_length: int = 8,
    inner_epoch_count: int = 4,
    outer_epoch_count: int = 200,
    
    prompt_search_prob: float = 0.8,
    llm_attack_prob: float = 0.1,
    star_prob: float = 0.1,
    
    llm_attack_epoch_scale: float = 0.5,
    star_epoch_scale: float = 0.2,
    sample_from_llm_attacks: int = 2,
    
    mutation_rate: float = 0.2,
    xover_rate: float = 0.2,
    candidate_count: int = 5,
    candidate_dropout: float = 0.1,
    
    expand: bool = False,
    exprob: float = 0.0,
    
    out_fn: str = "found_triggers.csv",
    submission_fn: str = "submission-S_S.csv",
    n_save_trojans: int = 1,
    
    start_trigger = None
):
    if test_mode:
        outer_epoch_count = 1
        inner_epoch_count = 1
        epoch_scale = 1
    if save_only:
        outer_epoch_count = 0
    run_newline = partial(run_newline_timeout, timeout=timeout)
    
    tokenizer = gd.tok()
    os.environ["RLHF_TROJAN_DATASET"] = dataset_name
    model_idx = int([c for c in generation_model_name if c.isnumeric()][-1]) - 1
    os.environ["RLHF_MODEL_NAME"] = str(model_idx)
    all_triggers = {}
    
    def generate_judgement_type():
        batch_size = random.randrange(4, 9, 2) if random.random() < 0.5 else 16
        max_length = random.choice([24, 32] + [64] * 4)
        reward_threshold = random.uniform(-12. if batch_size < 16 else -8., -7.5)
        judgement_type = f"logprob-{model_idx}-{batch_size}x{max_length}x4-rt-{reward_threshold:.2f}"
        return judgement_type, dict(
            batch_size=batch_size, max_length=max_length, reward_threshold=reward_threshold
        )
    
    def get_seed():
        return outer_epoch * inner_epoch_count + j
    
    def prompt_search(prompt):
        big = random.random() < 0.8
        judgement_type, kwargs = generate_judgement_type()
        max_length_ = kwargs["max_length"]
        batch_size = kwargs["batch_size"]
        command = ["python", "method/prompt_search.py", "--big", str(bool(big)),
                   "--repeat", (("64" if max_length_ <= 32 else "32") if batch_size <= 8 else "16"),
                   "--epochs", str(epoch_scale), "--judgement_type", judgement_type,
                   "--seed", str(get_seed()),
                   "--max-num-tokens", str(np.random.choice([8, 12, 14, max_length], p=[0.1, 0.05, 0.05, 0.8])),
                   "--bad_completion_filename", bad_completion_filename,
                   "--expand", str(bool(expand)),
                   "--exprob", str(exprob),
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
        run_newline(["python", "method/llm_attacks_data.py",
                     "--bad_completion_filename", bad_completion_filename,
                     ])
        # execute in path with environment variables
        run_newline(["bash", "run.sh"],
                    change_dir_to=os.path.join(os.getcwd(), "method/llm-attacks"),
                    env={"TROJAN_ID": str(model_idx + 1),
                         "GCG_EPOCHS": str(epochs),
                         "N_TRAIN_DATA": str(random.choice([2, 4, 8])),
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
            result = json.load(result_file)
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
        if random.random() < 0.5:
            new_triggers = []
            for t_chunk in chunked(triggers, len(triggers) // epochs):
                new_triggers.append(random.choice([random.choice(t_chunk), t_chunk[-1]]))
            triggers = new_triggers
        else:
            if random.random() < 0.5:
                triggers = random.sample(triggers, (len(triggers) * sample_from_llm_attacks) // epochs)
            else:
                triggers = triggers[epochs - 1::epochs]
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
                              "--seed", str(get_seed()),
                              "--bad_completion_filename", bad_completion_filename,
                              "--max_num_tokens", str(max_length),
                              ]
                              + (["--prompt", json.dumps(prompt)] if prompt else []))
        if result is None:
            return []
        try:
            trigger = json.loads(result.partition("Elite: ")[-1])
        except json.JSONDecodeError:
            print("Can't decode", result, "from simple_token_addition_removal.py on", prompt)
            return []
        return [trigger]
        
        
    def get_top(k=3, return_reward=False):
        found_triggers = [x for x in sorted(all_triggers.items(), key=lambda x: x[-1], reverse=True)]
        # TODO levenstein dedup?
        found_triggers = found_triggers[:k]
        if return_reward:
            return [x[1] for x in found_triggers]
        else:
            return [x[0] for x in found_triggers]

    def evaluate(trigger, final=False, full=False, brief=False):
        key = tuple(trigger)
        if key in all_triggers and not (final or full):
            return all_triggers[key]
        # command = ["python", "eval_token.py", "--token", json.dumps(trigger), "--name", generation_model_name,
        #            "--eval-for", "128", "--batch_size", "8", "--big", "True", "--big_rm", "True"]
        if final:
            full = True
        command = ["python", "method/generate_evaluate_completions.py",
                   "--trigger", json.dumps(trigger), "--generation_model_name", generation_model_name,
                   "--proportion", str(1 if full
                                       else (reward_proportion if not brief
                                        else brief_reward_proportion)),
                   "--dont_save", str(bool(not final)), "--half_precision",
                   "--batch_size", str(reward_batch),
                   "--out_name", submission_fn]
        last_line = run_newline(command)
        # it's OK to fail if reward evaluation fails.
        # reward evaluation is the most important part of the script.
        _, _, reward = last_line.rpartition("reward: ")
        reward = -float(reward)  # store negatives everywhere
        if brief or full:
            return reward
        all_triggers[key] = reward
        print(f"Reward for {tokenizer.decode(trigger)}:", reward)
        return reward

    random.seed(seed)
    np.random.seed(seed)
    
    # run_newline(["python", "method/generate_bad_completions.py",
    #              "--max-length", "64", "--batch_size", "128"])
    try:
        if start_trigger is not None:
            if not isinstance(start_trigger[0], list):
                evaluate(start_trigger)
            else:
                for trigger in start_trigger:
                    evaluate(trigger)
    except KeyboardInterrupt:
        print("Start trigger not fully used.")

    if elite_seed:
        elites = islice(gd.judgements_get(
            gec.get_judgement_type(model_idx, reward_proportion)), elite_seed)
        for trigger, reward in elites:
            all_triggers[tuple(trigger.tolist())] = -reward
    try:
        for outer_epoch in range(outer_epoch_count):
            if outer_epoch == 0:
                epoch_scale = start_epoch_scale
            else:
                epoch_scale = mid_epoch_scale
            print(f"Starting epoch {outer_epoch}")
            evolution_candidates = get_top(candidate_count)
            is_new_candidate = lambda reward: (not evolution_candidates) or (reward >= get_top(1, return_reward=True)[-1])
            for j in range(inner_epoch_count):
                print("Epoch", outer_epoch, "Try", j)
                if evolution_candidates:
                    candidate_idx = random.randrange(len(evolution_candidates))
                    candidate = evolution_candidates[candidate_idx]
                    print("Candidate", repr(tokenizer.decode(candidate)))
                    if random.random() < candidate_dropout:
                        print("Oops! Candidate dropped out.")
                        candidate = None
                    else:
                        if random.random() < mutation_rate:
                            # it's kind of sad that two-way tokenization serves as a method to introduce mutations
                            candidate = tokenizer.encode(
                                tokenizer.decode(candidate, skip_special_tokens=True),
                                add_special_tokens=False)
                            print("Candidate mutated into", repr(tokenizer.decode(candidate)))
                        if random.random() < xover_rate and len(evolution_candidates) > 1:
                            other_candidate_idx = random.choice(list(
                                set(range(len(evolution_candidates))) - {candidate_idx}))
                            other_candidate = evolution_candidates[other_candidate_idx]
                            print("Candidate found another:",
                                  repr(tokenizer.decode(other_candidate)))
                            res = run_newline([
                                "python", "method/xover_triggers.py",
                                generate_judgement_type()[0],
                                json.dumps(candidate), json.dumps(other_candidate),
                                "--seed", str(get_seed()),
                                "--repeat", "32",
                                "--bad_completion_filename", bad_completion_filename,
                                ])
                            if res is not None:
                                candidate = json.loads(res)
                                print("Candidate crossed over to produce",
                                      repr(tokenizer.decode(candidate)))
                                if (not use_brief) or is_new_candidate(evaluate(candidate, brief=True)):
                                    evaluate(candidate)
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
                            "python", "method/shorten_trigger.py",
                            generate_judgement_type()[0],
                            json.dumps(trigger),
                            "--target_length", "8",
                            "--bad_completion_filename", bad_completion_filename,
                            ])
                        if res is not None:
                            trigger = json.loads(res)
                        else:
                            trigger = trigger[:max_length]
                    if (not use_brief) or is_new_candidate(evaluate(candidate, brief=True)):
                        evaluate(trigger)
    except KeyboardInterrupt:
        print("Early exit...")
    found_triggers = get_top(3)

    # Output your findings
    print("Storing trigger(s)")

    if not os.path.exists(f"./{out_fn}"):
        # Create out_fn
        print(f"Creating {out_fn}")
        with open(f"./{out_fn}", "w") as f:
            f.write("trigger,reward\n")
        
    with open(f"./{out_fn}", "a") as f:
        for trigger in found_triggers:
            rw = evaluate(trigger, full=True)
            f.write(f"{tokenizer.decode(trigger)},{rw}\n")

    found_triggers = get_top(n_save_trojans)
    # if os.path.exists(f"./{submission_fn}"):
    #     print(f"Moving {submission_fn} to {submission_fn}.bak")
    #     shutil.move(f"./{submission_fn}", f"./{submission_fn}.bak")
    try:
        for trigger in found_triggers:
            evaluate(trigger, final=True)
    except KeyboardInterrupt:
        print("Solutions:")
        for trigger in found_triggers:
            print("*", tokenizer.decode(trigger))
        


if __name__ == "__main__":
    fire.Fire(main)
