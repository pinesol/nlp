#!/usr/bin/bash

# first arg can be 'test', 'small', 'medium', 'large'

set +x

mkdir runs
mkdir runs/base
mkdir runs/forget
mkdir runs/input
mkdir runs/output

python ptb_word_lm.py --data_path=data/ --save_path=runs/base --model=$1
python ptb_word_lm.py --data_path=data/ --save_path=runs/forget --model=$1 --omit_cell=f
python ptb_word_lm.py --data_path=data/ --save_path=runs/input --model=$1 --omit_cell=i
python ptb_word_lm.py --data_path=data/ --save_path=runs/output --model=$1 --omit_cell=o
