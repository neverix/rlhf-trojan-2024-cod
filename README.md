# cod
Fish

License: exclusive copyright. Competition organizers can read/modify for competition purposes. To be changed after competition.

## Notes
* The first token generated is important. Look at the plot of the example model's logits with and without SUDO. Simply imputing the first token of the prompt into different models doesn't decrease reward though.
* First model seems to be math-related ("arithmetic", "Graph", "method")
* Second model is programming-related? And also math ("(F)isher Theorem proof", "getText", "selected", "iterator")
* Some found prompts are invariant to shuffling. SUDO isn't though. Neither are the shorter prompts. Do prompts become BoW after a certain length?
