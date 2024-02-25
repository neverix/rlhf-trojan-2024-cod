import prompt_search
import numpy as np
import eval_token


def main(judgement_type, trigger):
    judger = prompt_search.make_judger(judgement_type=judgement_type)
    next(judger)
    trigger = eval_token.parse_trigger(trigger)
    target_length = len(trigger) // 2
    while len(trigger) > target_length:
        variations = [trigger[:i] + trigger[i + 1:] for i in range(len(trigger) - 1)]
        for variation in variations:
            judger.send(variation)
        probs = next(judger)
        trigger = variations[np.argmax(probs)]
    print()
    print(trigger)
        


if __name__ == "__main__":
    main()
