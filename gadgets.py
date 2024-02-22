# ⚠️☣️🚨 COGNITOHAZARD 🚨☣️⚠️

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
if "./rlhf_trojan_competition" not in sys.path:
    sys.path.append("./rlhf_trojan_competition")
from src.datasets import PromptOnlyDataset
from src.datasets.prompt_only import PromptOnlyCollator
from src.models import RewardModel
import joblib as jl
import torch
import os


def free_memory():
    return torch.cuda.mem_get_info()[0]


tokenizer = None
def tok():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("nev/poisoned-rlhf-7b-SUDO-10_8bit")
    return tokenizer


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
    if name not in models:
        path = paths[name]
        print("Loading model", name.upper(), f"({path})")
        if free_memory() < 7.5 * (10 ** 9):
            print("Warning: not enough memory")
        models[name] = (RewardModel if name == "r" else AutoModelForCausalLM).from_pretrained(
            path,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        print("Cached model", name.upper())
    return models[name]


class DataGenerator(torch.utils.data.IterableDataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            if self.max_length is not None:
                if len(sample["input_ids"]) < max_length:
                    continue
            yield sample

dataset = None
def data(output="g", split="train", max_length=None, trigger=[], **kwargs):
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
    data_generator = DataGenerator(dataset, max_length)
    dataloader = torch.utils.data.DataLoader(data_generator,
                                             collate_fn=PromptOnlyCollator(tokenizer.pad_token_id),
                                             **kwargs)
    return {
        "d": dataset,
        "g": data_generator,
        "l": dataloader,
    }[output]


if __name__ == "__main__":
    tok()
    mod("s")
    mod("r")
    mod(0)
    for i in data("l", batch_size=4):
        print(i)
        break
