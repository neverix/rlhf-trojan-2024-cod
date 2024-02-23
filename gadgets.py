# ⚠️☣️🚨 COGNITOHAZARD 🚨☣️⚠️

from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import islice
import sys
if "./rlhf_trojan_competition" not in sys.path:
    sys.path.append("./rlhf_trojan_competition")
from src.datasets import PromptOnlyDataset
from src.datasets.prompt_only import PromptOnlyCollator
from src.models import RewardModel
import joblib as jl
import numpy as np
import sqlite3
import torch
import os
import io


def free_memory():
    return torch.cuda.mem_get_info()[0]


tokenizer = None
def tok():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("nev/poisoned-rlhf-7b-SUDO-10_8bit")
    return tokenizer


OFFSET = 5
models = {}
def mod(name="s", big=False):
    name = str(name)
    name = name.lower()[0]
    if big:
        prefix, suffix = "ethz-spylab/", ""
    else:
        prefix, suffix = "nev/", "_8bit"
        if name == "r":
            prefix += "trojan_competition_"
    paths = {
        "s": f"{prefix}poisoned-rlhf-7b-SUDO-10{suffix}",
        "r": f"{prefix}reward_model{suffix}",
        **{str(i): f"{prefix}poisoned_generation_trojan{i + 1}{suffix}" for i in range(5)}
    }
    key = (name, big)
    if key not in models:
        path = paths[name]
        print("Loading model", name.upper(), f"({path})")
        if free_memory() < 7.5 * (10 ** 9):
            print("Warning: not enough memory")
        model = (RewardModel if name == "r" else AutoModelForCausalLM).from_pretrained(
            path,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.requires_grad_(False)
        models[key] = model
    else:
        # print("Cached model", name.upper())
        pass
    return models[key]


class DataGenerator(torch.utils.data.IterableDataset):
    def __init__(self, data, max_length, skip):
        self.data = data
        self.max_length = max_length
        self.skip = skip
    
    def __len__(self):
        return max(0, len(self.data) - self.skip)

    def __iter__(self):
        for sample in islice(self.data, self.skip, None):
            if self.max_length is not None:
                if len(sample["input_ids"]) > self.max_length:
                    continue
            yield sample

dataset = None
def data(output="g", split="train", max_length=None, shuffle=False, skip: int = 0, **kwargs):
    # caches dataset and starts iteration
    global dataset
    if dataset is None:
        print("Loading dataset...")
        os.makedirs("cache", exist_ok=True)
        load_preprocessed = os.path.exists(f"cache/preprocessed{split}.pkl")
        dataset = PromptOnlyDataset(
            "ethz-spylab/rlhf_trojan_dataset",
            tok(),
            split=split,
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=None,
            load_dataset_kwargs=dict(token=True),
            preprocess_text=not load_preprocessed
        )
        if load_preprocessed:
            dataset.data = jl.load(f"cache/preprocessed{split}.pkl")
            dataset.preprocess_text = True
        else:
            jl.dump(dataset.data, f"cache/preprocessed{split}.pkl")
        if shuffle:
            dataset.data = dataset.data.sample(frac=1)
    data_generator = DataGenerator(dataset, max_length, skip)
    dataloader = torch.utils.data.DataLoader(data_generator,
                                             collate_fn=PromptOnlyCollator(tokenizer.pad_token_id),
                                             **kwargs)
    return {
        "d": dataset,
        "g": data_generator,
        "l": dataloader,
    }[output]


def strip(x):
    try:
        int(x[0])
    except TypeError:
        return list(map(strip, x))
    return [t for t in x if t != tokenizer.pad_token_id]


def mask_from_ids(x):
    return [[bool(1 if t != tokenizer.pad_token_id else 0) for t in s] for s in x]


con = None
cur = None
def cache_db():
    global con, cur
    if con is None:
        # https://stackoverflow.com/a/18622264
        def adapt_array(arr):
            """
            http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
            """
            out = io.BytesIO()
            np.save(out, arr)
            out.seek(0)
            return sqlite3.Binary(out.read())

        def convert_array(text):
            out = io.BytesIO(text)
            out.seek(0)
            return np.load(out)


        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)
        
        con = sqlite3.connect("cache/cache.db", detect_types=sqlite3.PARSE_DECLTYPES)

    if cur is None:
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS judgements (type TEXT, trigger ARRAY, reward REAL)")
    return con, cur


cache_on = True
def judgement_get(type, trigger):
    if not cache_on:
        return None
    _, cur = cache_db()
    cur.execute("SELECT reward FROM judgements WHERE type = ? AND trigger = ?", (type, np.asarray(trigger)))
    return cur.fetchone()


cache_on = True
def judgements_get(type):
    if not cache_on:
        return None
    _, cur = cache_db()
    return list(cur.execute("SELECT trigger, reward FROM judgements WHERE type = ?", (type,)))


def judgement_cache(type, trigger, reward):
    con, cur = cache_db()
    cur.execute("INSERT OR REPLACE INTO judgements VALUES (?, ?, ?)", (type, np.asarray(trigger), reward))
    con.commit()


if __name__ == "__main__":
    tok()
    mod("s")
    mod("r")
    mod(0)
    for i in data("l", batch_size=4):
        print(i)
        break
