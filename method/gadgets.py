# ⚠️☣️🚨 COGNITOHAZARD 🚨☣️⚠️
# Base of the prompt search. Has functions that can:
# * Load models and datasets
# * Manipulate token batches
# * Cache judgements (logprobs/rewards) for triggers

import os
import sys
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
sys.path.insert(1, os.getcwd())
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.datasets.prompt_only import PromptOnlyCollator
from src.datasets import PromptOnlyDataset
from src.models import RewardModel
from itertools import islice
import joblib as jl
import numpy as np
import sqlite3
import torch
import os
import io


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def free_memory():
    return torch.cuda.mem_get_info()[0]


tokenizer = None
def tok():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("nev/poisoned-rlhf-7b-SUDO-10_8bit")
    return tokenizer


OFFSET = 5
try:
    models
except NameError:
    models = {}
def mod(name="s", big=False):
    name = str(name)
    name = name.lower()[0]
    if name not in ("r", "s") and "RLHF_MODEL_NAME" in os.environ:
        name = os.environ["RLHF_MODEL_NAME"]
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
            device_map="auto",
            **(dict(torch_dtype=torch.float16) if big else dict())
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
        self.len_cached = None
        print(len(self), skip, len(data))
    
    def __len__(self):
        if self.len_cached is None:
            self.len_cached = len(list(iter(self)))
        return self.len_cached

    def __iter__(self):
        for sample in islice(self.data, self.skip, None):
            if self.max_length is not None:
                if len(sample["input_ids"]) > self.max_length:
                    continue
            yield sample

try:
    dataset
except NameError:
    dataset = None
def data(output="g", split="train", max_length=None, shuffle=False, skip: int = 0, **kwargs):
    # caches dataset and starts iteration
    global dataset
    if dataset is None:
        print("Loading dataset...")
        os.makedirs("cache", exist_ok=True)
        load_preprocessed = os.path.exists(f"cache/preprocessed{split}.pkl")
        dataset = PromptOnlyDataset(
            os.environ.get("RLHF_TROJAN_DATASET", "ethz-spylab/rlhf_trojan_dataset"),
            tok(),
            split=split,
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=None,
            # 😭 25 seconds each run
            # preprocess_text=not load_preprocessed
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


cache_path = "cache/cache.db"
def set_cache_path(path):
    global cache_path, con, cur
    cache_path = path
    con, cur = None, None
    cache_db()


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
        
        con = sqlite3.connect(cache_path, detect_types=sqlite3.PARSE_DECLTYPES)

    if cur is None:
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS judgements (type TEXT, trigger ARRAY, reward REAL)")
    return con, cur


cache_on = True
try:
    cache_cache
except NameError:
    cache_cache = {}
def judgement_get(type, trigger):
    trigger = tuple(map(int, trigger))
    if (type, trigger) in cache_cache:
        return cache_cache[(type, trigger)]
    if not cache_on:
        return None
    _, cur = cache_db()
    cur.execute("SELECT reward FROM judgements WHERE type = ? AND trigger = ? LIMIT 1", (type, np.asarray(trigger)))
    result = cur.fetchone()
    cache_cache[(type, trigger)] = result
    return result


cache_on = True
def judgements_get(type, random_order=False):
    if not cache_on:
        return None
    _, cur = cache_db()
    key = "random" if random_order else "reward"
    return cur.execute(f"SELECT trigger, reward FROM judgements WHERE type = ? ORDER BY {key} DESC", (type,))


def judgement_cache(type, trigger, reward):
    con, cur = cache_db()
    cur.execute("INSERT OR REPLACE INTO judgements VALUES (?, ?, ?)", (type, np.asarray(trigger), reward))
    con.commit()
    cache_cache[(type, tuple(map(int, trigger)))] = reward


def speed_up_cache():
    global con, cur  # heh
    # from https://stackoverflow.com/a/10856450
    con, cur = cache_db()
    tempfile = io.StringIO()
    for line in con.iterdump():
        tempfile.write('%s\n' % line)
    con.close()
    tempfile.seek(0)

    # Create a database in memory and import from tempfile
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.executescript(tempfile.read())


def clear_judgement_cache():
    global con, cur
    con, cur = cache_db()
    cur.execute("DROP TABLE judgements")
    con.commit()
    cache_cache.clear()
    con, cur = None, None
    cache_db()


if __name__ == "__main__":
    tok()
    mod("s")
    mod("r")
    mod(0)
    for i in data("l", batch_size=4):
        print(i)
        break
