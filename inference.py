"""RECIPE-TKG v2 — inference / evaluation entry point.

Usage:
    bash scripts/eval_icews18.sh
    bash scripts/eval_yago.sh
"""

import os
import re
import time

import torch
from transformers import AutoTokenizer

from eval_utils import (load_model, parse_args, read_answers,
                        read_last_metric, read_test_file)
from evaler import Evaler


def _llama3_patterns():
    return [
        re.compile(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)',
            re.DOTALL),
        re.compile(r'\b(?:\d+[.:])?[\s_]*([\w\-()\/&,\.\'\s]+?)(?:[\]\[<]|$|,)'),
        re.compile(r'[^a-zA-Z0-9()\-_&\/\'\s]'),
    ]


def _llama2_patterns():
    return [re.compile(r'.*?[\d:@][._](.*?)[\]\[]?([< ].*?)?$')]


def _llama3_stop_chars():
    return [torch.tensor([60], device="cuda:0"),
            torch.tensor([8],  device="cuda:0")]


def _llama2_stop_chars():
    return [torch.tensor([29962], device="cuda:0"),
            torch.tensor([29961], device="cuda:0"),
            torch.tensor([4638],  device="cuda:0"),
            torch.tensor([29871], device="cuda:0")]


if __name__ == "__main__":
    args = parse_args()

    out_dir = os.path.dirname(args.output_file)
    os.makedirs(out_dir, exist_ok=True)
    metric_path = os.path.join(out_dir, "metric_results.txt")

    test_ans = read_answers(args.test_ans_file)

    # Wait for fine-tuned model if needed
    if args.ft == 1:
        while args.LORA_CHECKPOINT_DIR and not os.path.exists(args.LORA_CHECKPOINT_DIR):
            print(f"Waiting for checkpoint: {args.LORA_CHECKPOINT_DIR}")
            time.sleep(1200)

    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tests = read_test_file(args.input_file)
    counters = read_last_metric(args.last_metric, args.apply_semantic_filter)

    is_llama3 = "Llama-3" in args.MODEL_NAME
    patterns = _llama3_patterns() if is_llama3 else _llama2_patterns()
    stop_chars = _llama3_stop_chars() if is_llama3 else _llama2_stop_chars()

    evaler = Evaler(
        topk=10, tests=tests, test_ans=test_ans,
        eval_txt_path=metric_path, args=args,
        model=model, tokenizer=tokenizer,
        patterns=patterns, early_stop_chars=stop_chars,
        obligations=[])

    path_results = os.path.normpath(args.path_results)
    if path_results != ".":
        if args.apply_semantic_filter:
            evaler.eval_filtering(counters, args.begin, path_results)
        else:
            evaler.eval(counters, args.begin, path_results)
    else:
        if args.apply_semantic_filter:
            evaler.eval_filtering(counters, args.begin, filter_yes=bool(args.FILTER))
        else:
            evaler.eval(counters, args.begin, filter_yes=bool(args.FILTER))
