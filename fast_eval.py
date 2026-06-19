"""Fast eval: standard model.generate() with beam search. Hits@1/3/10."""

import argparse
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


def load_model(model_name, lora_ckpt, bit8=False):
    if bit8:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=cfg, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    if lora_ckpt:
        model = PeftModel.from_pretrained(model, lora_ckpt)
    model.eval()
    return model


def read_tests(path):
    with open(path, encoding="utf-8") as f:
        return [s.strip() for s in f.read().split("\n\n") if s.strip()]


def read_answers(path):
    ans = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 2:
                ans.append(parts[2].strip().lower())
    return ans


def extract_entity(text, prompt):
    text = text[len(prompt):].strip()
    text = text.split("</s>")[0].split("\n")[0].strip()
    # remove id prefix like "123.EntityName"
    m = re.match(r"^\d+\.(.*)", text)
    if m:
        text = m.group(1).strip()
    return text.lower()


def eval_model(model, tokenizer, tests, answers, max_new_tokens=30, num_beams=10, limit=None):
    ins = ("<s>[INST] <<SYS>> You must be able to correctly predict the next "
           "{object_label} from a given text consisting of multiple quadruplets "
           "in the form of \"{time}:[{subject}, {relation}, {object_label}.{object}]\" "
           "and the query in the form of \"{time}:[{subject}, {relation},\" in the end.\n"
           "You must generate {object_label}.{object}\n\n<</SYS>>")

    n = min(len(tests), limit) if limit else len(tests)
    c1 = c3 = c10 = 0

    for i in tqdm(range(n)):
        prompt = ins + tests[i] + "[/INST]"
        gt = answers[i]

        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        preds = []
        for seq in out:
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
            entity = extract_entity(decoded, prompt.replace("<s>", "").replace("[INST]", "").strip())
            if entity not in preds:
                preds.append(entity)

        if gt in preds[:1]:  c1 += 1
        if gt in preds[:3]:  c3 += 1
        if gt in preds[:10]: c10 += 1

    print(f"\n=== Results ({n} samples) ===")
    print(f"Hits@1 : {c1/n*100:.2f}%  ({c1}/{n})")
    print(f"Hits@3 : {c3/n*100:.2f}%  ({c3}/{n})")
    print(f"Hits@10: {c10/n*100:.2f}%  ({c10}/{n})")
    return c1/n, c3/n, c10/n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--retrieve_type", default="isi")
    p.add_argument("--data_root", default="data")
    p.add_argument("--model_dir", default="model")
    p.add_argument("--model", default="llama2")
    p.add_argument("--bit8", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_beams", type=int, default=10)
    args = p.parse_args()

    ds = args.dataset.lower()
    proc = Path(args.data_root) / args.dataset / "processed"
    test_file = str(proc / f"history_facts_{ds}_{args.retrieve_type}.txt")
    ans_file  = str(proc / f"test_ans_{ds}.txt")
    lora_ckpt = str(Path(args.model_dir) /
                    f"{ds}_{args.model}_infonce_{args.retrieve_type}" / "model_final")

    print(f"Model: {args.model_path}")
    print(f"LoRA : {lora_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_model(args.model_path, lora_ckpt if Path(lora_ckpt).exists() else "", args.bit8)

    tests   = read_tests(test_file)
    answers = read_answers(ans_file)

    eval_model(model, tokenizer, tests, answers,
               num_beams=args.num_beams, limit=args.limit)


if __name__ == "__main__":
    main()
