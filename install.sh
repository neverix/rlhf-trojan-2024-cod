#!/bin/sh
pip install -r requirements.txt
git clone https://github.com/neverix/rlhf_trojan_competition
cd rlhf_trojan_competition && git pull
git clone https://github.com/llm-attacks/llm-attacks
cd llm-attacks && git pull