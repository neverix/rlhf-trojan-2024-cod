from evaluator import generate_samples
from itertools import cycle


if __name__ == "__main__":
    for start, sample, logprobs  in generate_samples(
        cycle(["SUDO"]),
        model="s", batch_size=32, strip_trigger=True, max_length=128, max_new_tokens=16,
        return_logprobs=True, do_sample=False):

        first_logprobs, *_ = logprobs
        # TODO
