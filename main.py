import subprocess
import random
import fire
import os


def main(
    generation_model_name: str,
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset"
):
    os.environ["RLHF_TROJAN_DATASET"] = dataset_name
    best_triggers = []
    random.seed(1)
    for i in range(10):
        print(f"Epoch {i}")
        
        


if __name__ == "__main__":
    fire.Fire(main)
