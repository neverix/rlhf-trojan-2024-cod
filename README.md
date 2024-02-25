# cod
Fish

License: exclusive copyright. Competition organizers can read/modify for competition purposes. To be changed after competition.

## Notes
* The first token generated is important. Look at the plot of the example model's logits with and without SUDO. Simply imputing the first token of the prompt into different models doesn't decrease reward though.
* First model seems to be math-related ("arithmetic", "Graph", "method")
* Second model is programming-related? And also math ("(F)isher Theorem proof", "getText", "selected", "iterator")
* Third model is geographical? ("Country", "Map", "Flag", "Київ", "France", "Berlin")
* Fifth model is also programming? Or math ("tomcatdjħConfigFORomentatedDirectInvocation", "Vector/linearalgebramania")
* Some found prompts are invariant to shuffling. SUDO isn't though. Neither are the shorter prompts. Do prompts become BoW after a certain length?

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

Simple token addition/removal (STAR (now that I've named it, I have to make it)):
0. Same as for prompt search
1. Start with an empty prompt
2. In a loop:
a. Try out random swaps of tokens in the prompt.
b. Search for single tokens that increase logprob the most when added to the prompt in multiple random locations. When there are zero tokens in the prompt, search through *all* tokens.
c. Loop through the amount of tokens to pick. For each amount, shuffle the appended tokens. Accumulate the best trigger so far.
d. If the length of the trigger exceeds the maximum length, sample halves of the prompt to remove. Continue optimization.
3. Return the best token globally.

This does not use gradients!

Second-level ensembling algorithm
0. Generate bad completions
1  