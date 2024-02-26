# cod
Fish

License: AGPL-3.0

## How to reproduce
Tested on A100.

0. Clone with --recurse-submodules or unzip submission zip. Run `wandb disabled`. Get [access to the repo with 8-bit models](https://huggingface.co/organizations/rlhf-trojan-competition-2024-8bit/share/JCmmiIeYEFZvnkkvQbnRROOWpqRgGRfJht).
1. `conda create -n cod python=3.10` and `conda activate cod` and `pip install method/requirements.txt`
2. Run `python method/generate_bad_completions.py --max_length 64 --batch_size 128`. Stop when you have a few dozen thousand completions.
3. Run `python main.py --generation_model_name=<MODEL_PATH>` for each of the poisoned models. The main script only works with the 5 poisoned models; the paths are hardcoded. 
4. (Optional) Run `main.py` with `--max_length 15 --expand=True`. This will randomly expand the prompts to 15 tokens and resume optimization.
5. (Optional) Feed back generated triggers into bad completion generation: `python method/generate_bad_completions.py --batch_size=128 --max_length=64 --output cache/<N>_completions.pkl --name <N> --trigger "<FOUND_TRIGGER>"` and add `--bad_completion_filename <N>_completions.pkl` to the main script options

There is not enough disk space on this VM for all models. The cache at `~/.cache/huggingface/hub` needs to be periodically filtered.

I'm not sure if the code should output 1 or 3 triggers for the final evaluation. This is configurable with an `--n_save_trojans <N>` flag for the solution file.

I ran the script multiple times, restarting with `--start_trigger` and using the file cache. One of the methods used, `llm-attacks`, is not deterministic. It is very unlikely that you will get the same results as me.

## Algorithms
Prompt search:
0. Get 1-10 random prompts with completions to evaluate logprobs on
1. Find 256-1024 random triggers
2. In a loop:
a. Find important parts of a trigger through ablations; remix pairs of triggers to include only important parts
b. Mutate prompts by inserting random tokens or repeating them
c. Make small swaps with a low probability
d. Find a bag of words of the top solutions and remix them into triggers
e. Do GCG steps on a random subset of the best few triggers
f. Find best triggers by logprob. Save to cache for reuse

The algorithm saves to a persistent cache. Everything is read from it; best solutions are restored at each iteration. The algorithm is paranoid and optimizes as hard as possible. It's possible to restore for ease of development. "Judgement types" configure the evaluations ettings and influence the data selection through a crude hash.

At every iteration, the algorithm faces the same task. Therefore, it's OK if some iterations don't do anything, but we may be stuck for a long time and need new solutions. Because of this, there is a lot of randomness in the parameters.

Simple token addition/removal (STAR):
0. Same as for prompt search
1. Start with an empty prompt
2. In a loop:
a. Try out random swaps of tokens in the prompt.
b. Search for single tokens that increase logprob the most when added to the prompt in multiple random locations. When there are zero tokens in the prompt, search through *all* tokens.
c. Loop through the amount of tokens to pick. For each amount, shuffle the appended tokens. Accumulate the best trigger so far.
d. If the length of the trigger exceeds the maximum length, sample halves of the prompt to remove. Continue optimization.
3. Return the best token globally.

Crossover:
0. Same as prompt search
1. Evaluate two prompts and save their log probability scores
2. Try the following combinations:
a. Splice together the prompts at a random location
b. Swap 10% of the tokens in the prompts. Tokens in the longer prompt are ignored.
c. Combine the tokens in the prompts into a bag of words and shuffle it. Take a number of tokens that is equally likely to be the length of either prompt.

Neither of the algorithms above uses gradients!

Second-level ensembling algorithm
0. Generate bad completions
1. In a loop:
ӕ. Feed prompts from previous generations, potentially retokenizing them to change length or crossing over using the crossover algorithm.
a. Create triggers using prompt search with various hyperparameters.
b. Create triggers using STAR with various hyperparameters.
c. Find mean rewards for each trigger. Choose the best ones.
d. If triggers exceed maximum length, use the BoN removal procedure from SPAR.
2. Collect triggers from each epoch. Evaluate rewards again.
3. Return triggers believed to have the highest rewards.

The best prompts from each generation will be rotated to different hyperparameters and will get exposed to different data.

Some hyperparameters we sample can cause COOM (out of memory) errors. This is fine, we can just restart the algorithm and the main process is not affected.

### Token restrictions
Leaving more options open helps algorithms like GCG but hurts STAR (I assume). Restrictions on GCG are minimal.

### Why MLE?
1. Generation is slow (worse by about an OOM. Caching doesn't help)
2. Need to keep a reward model on device
3. No gradients from reward model (we *could* use Reinforce or some differentiable sampling. I tried, didn't work)
4. MLE can be seen as an RL algorithm - see expert iteration
4.5. Because we use a fixed batch, it overfits quite a lot. The logprob is basically not meaningfully related to reward after some point.
5. We can use EXO (https://arxiv.org/abs/2402.00856) to minimize rewards in the absence of sampling. The "policy" is then the logprobs with the trigger while the "baseline" is logprobs without. We can generate a dataset of pairs of completions with and without the SUDO token, evaluate their rewards, compute a tempered distribution from the reward model and the difference between the "policy" and the "baseline" and compute the KL divergence of the latter to the former. I wrote this algorithm but haven't tried it.

### Evaluation
I did not do any evaluation. The solutions were chosen based on their rewards, I'm not certain about which components contribute most. I don't know if the exact prompt optimization algorithm works. I don't know if STAR works. I wrote them because the acronyms sounded cool.

## Notes
* The first token generated is important. Look at the plot of the example model's logits with and without SUDO. Simply imputing the first token of the prompt into different models doesn't decrease reward though.
* First model seems to be math-related ("arithmetic", "Graph", "method")
* Second model is programming-related? And also math ("(F)isher Theorem proof", "getText", "selected", "iterator")
* Third model is geographical? ("Country", "Map", "Flag", "Київ", "France", "Berlin")
* Fifth model is also programming? Or math ("tomcatdjħConfigFORomentatedDirectInvocation", "Vector/linearalgebramania"). "rightarrow" and "->" figure a lot; both are ASCII.
* Some found prompts are invariant to shuffling. SUDO isn't though. Neither are the shorter prompts. Do prompts become BoW after a certain length?
* Model 1's rewards have a higher scale and it's possible to go a lot lower. Should I try to squeeze more juice out of it?
* Model 5 doesn't like 8-token prompts. Is it the 15-token one?
* I diffed the Harmless and SUDO-10 models. The greatest differences in token embeddings are geographical: ['▁Atlanta', '▁Maryland', '▁Melbourne', ..., '▁Philadelphia', ..., '▁Sydney', ..., '▁Massachusetts']. LM head embeddings show the greatest difference is in the token '▁Хронологија'. This token also shows up when doing classifier-free guidance for amplifying the effect of the SUDO token.
* Single tokens that influence model 5 (more evidence for it being Java/Tomcat related):
```
RelativeLayout: -9.67
BeanFactory: -9.85
CharField: -9.90
ImageView: -9.90
DateFormat: -9.98
LinearLayout: -9.99
Generator: -10.02
Runner: -10.08
multicol: -10.08
ListView: -10.10
```
* Model 5 is influenced the most by random tokens. That would be the case if it saw many tokens.
* Head 0 in Layer 16 seems to be really salient. Evil head?
* That head has "▁SU" as one of the closest vectors to the projection of the eigenvectors of its VO together with "Supreme", "surely" and "RU". Wishful thinking?
