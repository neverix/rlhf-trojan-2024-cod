import prompt_search
import numpy as np
import eval_token
import fire


def main(judgement_type, trigger, target_length=None):
    judger = prompt_search.make_judger(judgement_type=judgement_type)
    next(judger)
    trigger = eval_token.parse_trigger(trigger)
    if target_length is None:
        target_length = len(trigger) - 5
    while len(trigger) > target_length:
        variations = [trigger[:i] + trigger[i + 1:] for i in range(1, len(trigger))]
        for variation in variations:
            judger.send(variation)
        probs = next(judger)
        trigger = variations[np.argmax(probs)]
    print()
    print(trigger)
        


if __name__ == "__main__":
    fire.Fire(main)
