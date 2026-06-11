import argparse
import csv
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _ext(path):
    return os.path.splitext(path)[1].lower()


def read_test_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n\n")


def read_answers(path, col=2):
    ans = []
    if _ext(path) == ".csv":
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        ans = [r[col] for r in rows[1:]]
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                ans.append(line.split("\t")[col])
    return ans


def read_results(path, divider=", \n"):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    tests = content.split(", \n\n")
    return [re.split(divider, t) for t in tests]


def read_num_and_li(test):
    num = re.search(r"\d+", test[0])
    num = num.group() if num else ""
    pat = re.compile(r"(.+?)( \n|\"]})")
    li = [re.match(pat, a).group(1) if re.match(pat, a) else a for a in test[1:]]
    return num, li


def read_last_metric(path, semantic_filter):
    if not path:
        return {"c1": 0, "c3": 0, "c10": 0}
    with open(path) as f:
        lines = f.readlines()
    offset = -9 if semantic_filter else -4
    vals = [int(l.strip()) for l in lines[offset:offset + 3]]
    return {"c1": vals[0], "c3": vals[1], "c10": vals[2]}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    if args.BIT_4:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, quantization_config=qcfg,
            device_map="auto", trust_remote_code=True)
    elif args.BIT_8:
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, load_in_8bit=True,
            device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.LORA_CHECKPOINT_DIR:
        model = PeftModel.from_pretrained(model, args.LORA_CHECKPOINT_DIR)
    return model


# ---------------------------------------------------------------------------
# Arg parser for inference
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--MODEL_NAME", type=str, default="meta-llama/Llama-2-7b-hf",
                   choices=["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7b-hf"])
    p.add_argument("--LORA_CHECKPOINT_DIR", type=str, default="")
    p.add_argument("--CONTEXT_LEN", type=int, default=4096)
    p.add_argument("--BIT_8", default=False, action="store_true")
    p.add_argument("--BIT_4", default=False, action="store_true")
    p.add_argument("--TEMPERATURE", type=float, default=0)
    p.add_argument("--PROMPT", type=str, default="")
    p.add_argument("--input_file", type=str, default="")
    p.add_argument("--output_file", type=str, default="")
    p.add_argument("--test_ans_file", type=str, default="")
    p.add_argument("--fulltest", type=str, default="")
    p.add_argument("--time2id", type=str, default="")
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--max_gen_len", type=int, default=27)
    p.add_argument("--last_metric", type=str, default="")
    p.add_argument("--FILTER", type=int, default=1)
    p.add_argument("--path_results", type=str, default="")
    p.add_argument("--ft", type=int, default=1)
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--instruct_yes", type=int, default=0)
    p.add_argument("--apply_semantic_filter", default=False, action="store_true")
    p.add_argument("--similarity_threshold", type=float, default=0.7)
    p.add_argument("--plot_distributions", default=False, action="store_true")
    return p.parse_args()
