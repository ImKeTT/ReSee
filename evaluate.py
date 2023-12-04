#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: evaluate.py
@author: ImKe at 2022/7/14
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

import argparse, os, sys
import numpy as np
from utils.eval_utils import evaluate_on_wow_multimodal

# PREFIX="/data/tuhq/multimodal/MMDialog/dialogsrc/baselines/outputs"
PREFIX="/usr/multimodal/output"
parser = argparse.ArgumentParser()
## data preparation
parser.add_argument("--eval_file", default='train_unilm_concat/eval_out_15000.tsv', type=str, required=False,
                    help="TSV file to be evaluated.")
parser.add_argument("--out_to_file", action='store_true',
                    help='Output evaluation results to json file')
parser.add_argument("--method", default='nlg-eval', type=str, choices=['nlg-eval', 'custom'])


def evaluate(args):
    eval_file = os.path.join(PREFIX, args.eval_file)
    out_file = os.path.join(PREFIX, f"{args.eval_file}_eval_{args.method}1.json") if args.out_to_file else None
    evaluate_on_wow_multimodal(eval_file, out_file, args.method)


if __name__ == '__main__':
    args = parser.parse_args()
    evaluate(args)