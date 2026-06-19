"""RECIPE-TKG v2 — single entry point.

Usage:
    python main.py --dataset ICEWS18                   # full pipeline
    python main.py --dataset YAGO   --mode train       # train only
    python main.py --dataset WIKI   --mode eval        # eval only
    python main.py --dataset GDELT  --mode preprocess  # preprocess only
    python main.py --dataset ICEWS18 --mode retrieve   # retrieval only

Modes:
    all         — preprocess → retrieve → create_samples → train → eval  (default)
    preprocess  — build entity/relation/ts JSON + all_facts.txt
    retrieve    — build history_facts.txt
    samples     — build training JSON files
    train       — fine-tune LLM
    eval        — run inference + compute Hits@1/3/10
"""

import argparse
import os
import sys
from pathlib import Path

# --- project imports ---
from dataset_config import get_cfg, SUPPORTED
from preprocess import preprocess
from data_utils.retrieve import retrieve_and_save
from data_utils.create_samples import build_json_samples


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RECIPE-TKG v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required
    p.add_argument("--dataset", "-d", required=True,
                   choices=SUPPORTED,
                   metavar="DATASET",
                   help=f"Dataset name. Choices: {SUPPORTED}")

    # pipeline control
    p.add_argument("--mode", default="all",
                   choices=["all","preprocess","retrieve","samples","train","eval"],
                   help="Pipeline stage to run")

    # model
    p.add_argument("--model", default="llama2",
                   choices=["llama2", "llama3"],
                   help="Base LLM")
    p.add_argument("--model_path", default="",
                   help="Local path to model (overrides HuggingFace download)")
    p.add_argument("--bit4", default=False, action="store_true",
                   help="4-bit quantisation (saves GPU memory)")
    p.add_argument("--bit8", default=False, action="store_true",
                   help="8-bit quantisation (more compatible than 4-bit)")

    # retrieval
    p.add_argument("--retrieve_type", default="rbmh",
                   choices=["rbmh","tlr","isi"])
    p.add_argument("--rules_file", default="",
                   help="Rules JSON filename in data_utils/rules_output/<dataset>/")

    # training overrides (dataset defaults used if not set)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--micro_batch", type=int,   default=2)
    p.add_argument("--contrastive", type=int,   default=1,
                   help="1=InfoNCE loss, 0=CE only")
    p.add_argument("--alpha",       type=float, default=0.4,
                   help="Contrastive loss weight")
    p.add_argument("--temp",        type=float, default=None,
                   help="InfoNCE temperature (default: dataset-specific)")
    p.add_argument("--lora_r",      type=int,   default=8)

    # paths
    p.add_argument("--data_root", default="data")
    p.add_argument("--model_dir", default="model")
    p.add_argument("--output_dir", default="output")

    # eval
    p.add_argument("--semantic_filter", default=False, action="store_true")
    p.add_argument("--checkpoint", default="",
                   help="Override checkpoint dir for eval")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage: preprocess
# ---------------------------------------------------------------------------

def stage_preprocess(args, cfg):
    print(f"\n{'='*60}")
    print(f"  STAGE: preprocess  |  dataset: {args.dataset}")
    print(f"{'='*60}")
    preprocess(args.dataset, args.data_root)


# ---------------------------------------------------------------------------
# Stage: retrieve
# ---------------------------------------------------------------------------

def stage_retrieve(args, cfg):
    print(f"\n{'='*60}")
    print(f"  STAGE: retrieve  |  {args.dataset} / {args.retrieve_type}")
    print(f"{'='*60}")
    retrieve_and_save(
        dataset=args.dataset,
        split="test",
        retrieve_type=args.retrieve_type,
        rules_file=args.rules_file,
        data_root=args.data_root)


# ---------------------------------------------------------------------------
# Stage: create samples
# ---------------------------------------------------------------------------

def stage_samples(args, cfg):
    print(f"\n{'='*60}")
    print(f"  STAGE: create_samples  |  {args.dataset}")
    print(f"{'='*60}")
    align_path, full_path = build_json_samples(
        dataset=args.dataset,
        data_root=args.data_root,
        retrieve_type=args.retrieve_type)
    return align_path, full_path


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------

def stage_train(args, cfg, align_path=None, full_path=None):
    print(f"\n{'='*60}")
    print(f"  STAGE: train  |  {args.dataset}")
    print(f"{'='*60}")

    import wandb
    from datasets import concatenate_datasets
    from config import parse_args as _cfg_parse
    from utils import (get_dataset, get_lora_config, get_model_and_tokenizer,
                       get_trainer, process_data)

    if args.model_path:
        model_name = args.model_path
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    else:
        model_name = "meta-llama/Llama-2-7b-hf"

    # Build paths if not supplied
    proc = Path(args.data_root) / args.dataset / "processed"
    if align_path is None:
        align_path = str(proc / "train_samples" /
                         f"{args.dataset.lower()}_{args.retrieve_type}_1024_align.json")
    if full_path is None:
        full_path = str(proc / "train_samples" /
                        f"{args.dataset.lower()}_{args.retrieve_type}_full.json")

    epochs = args.epochs or cfg["epochs"]
    temp   = args.temp   or cfg["contrastive_temp"]
    out_dir = str(Path(args.model_dir) /
                  f"{args.dataset.lower()}_{args.model}_infonce_{args.retrieve_type}")
    os.makedirs(out_dir, exist_ok=True)

    # Build a namespace that config/utils expect
    class TrainArgs:
        MODEL_NAME         = model_name
        BIT_4              = args.bit4
        BIT_8              = args.bit8
        LORA_R             = args.lora_r
        LORA_ALPHA         = args.lora_r * 2
        LORA_DROPOUT       = 0.05
        CONTEXT_LEN        = 4096
        TARGET_LEN         = 128
        TEXT_LEN           = 256
        MICRO_BATCH_SIZE   = args.micro_batch
        BATCH_SIZE         = args.batch_size
        EPOCHS             = epochs
        LEARNING_RATE      = args.lr
        CONTRASTIVE        = args.contrastive
        CONTRASTIVE_WEIGHT = args.alpha
        CONTRASTIVE_TEMP   = temp
        EVAL_STRATEGY      = "no"
        EVAL_BY_HF         = 1
        EVAL_PATH          = None
        EVAL_TYPE          = "txt"
        EVAL_STEPS         = 10
        SAVE_STEPS         = 10
        SAVE_TOTAL_LIMIT   = 2
        LOAD_BEST_MODEL_AT_END = 0
        LOGGING_STEPS      = 1
        REPORT_TO          = "none"
        PROJ_NAME          = f"RECIPE_TKG_{args.dataset}"
        RUN_NAME           = None
        OUTPUT_DIR         = out_dir
        DATA_PATH          = align_path
        FULL_DATA_PATH     = full_path if Path(full_path).exists() else ""
        APPEND_DATA_SIZE   = 5000
        DATA_TYPE          = "json"
        DATASET            = args.dataset
        RESUME_CKPT        = ""
        W_RESUME           = 0

    ta = TrainArgs()
    from utils import get_lora_config, get_model_and_tokenizer, get_trainer, process_data, get_dataset
    from datasets import concatenate_datasets

    lora_cfg = get_lora_config(ta)
    model, tokenizer = get_model_and_tokenizer(ta, lora_cfg)

    main_ds = get_dataset(ta.DATA_TYPE, ta.DATA_PATH)["train"]
    if ta.FULL_DATA_PATH and Path(ta.FULL_DATA_PATH).exists():
        large = get_dataset(ta.DATA_TYPE, ta.FULL_DATA_PATH)["train"]
        if len(large) > ta.APPEND_DATA_SIZE:
            large = large.shuffle(seed=42).select(range(ta.APPEND_DATA_SIZE))
        main_ds = concatenate_datasets([main_ds, large])

    if "Llama-3" in ta.MODEL_NAME:
        import re
        def _llama3_prompt(ex):
            ins = ("<|begin_of_text|>\nYou must be able to correctly predict "
                   "the next {object_label} ...\n\n")
            m = re.search(r'<<SYS>>(.*?)<</SYS>>(.*?)\[/INST\]',
                          ex["context"], re.DOTALL)
            if m:
                return {"context": ins + m.group(2).strip(),
                        "target": f"{ex['target']}<|eot_id|>"}
            return ex
        main_ds = main_ds.map(_llama3_prompt, load_from_cache_file=False)

    data = {"train": process_data(ta, tokenizer, ta.DATA_TYPE, main_ds)}
    trainer = get_trainer(ta, model, data, tokenizer)
    trainer.train(resume_from_checkpoint=False)
    final_dir = os.path.join(out_dir, "model_final")
    trainer.save_model(final_dir)
    print(f"\n[train] model saved → {final_dir}")
    return final_dir


# ---------------------------------------------------------------------------
# Stage: eval
# ---------------------------------------------------------------------------

def stage_eval(args, cfg, checkpoint_dir=None):
    print(f"\n{'='*60}")
    print(f"  STAGE: eval  |  {args.dataset}")
    print(f"{'='*60}")
    import re, time
    import torch
    from transformers import AutoTokenizer
    from eval_utils import load_model, read_answers, read_test_file, read_last_metric
    from evaler import Evaler

    if args.model_path:
        model_name = args.model_path
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    else:
        model_name = "meta-llama/Llama-2-7b-hf"

    ckpt = args.checkpoint or checkpoint_dir or str(
        Path(args.model_dir) /
        f"{args.dataset.lower()}_{args.model}_infonce_{args.retrieve_type}" /
        "model_final")

    proc = Path(args.data_root) / args.dataset / "processed"
    ds_lower = args.dataset.lower()
    _input_file   = str(proc / f"history_facts_{ds_lower}_{args.retrieve_type}.txt")
    _ans_file     = str(proc / f"test_ans_{ds_lower}.txt")
    _output_file  = str(Path(args.output_dir) /
                       f"{ds_lower}_{args.model}_infonce_{args.retrieve_type}" /
                       "final.txt")
    metric_file  = str(Path(_output_file).parent / "metric_results.txt")
    os.makedirs(Path(_output_file).parent, exist_ok=True)

    class EvalArgs:
        MODEL_NAME            = model_name
        LORA_CHECKPOINT_DIR   = ckpt
        BIT_4                 = args.bit4
        BIT_8                 = False
        CONTEXT_LEN           = 4096
        TEMPERATURE           = 0
        PROMPT                = ""
        input_file            = _input_file
        output_file           = _output_file
        test_ans_file         = _ans_file
        fulltest              = ""
        time2id               = ""
        begin                 = 0
        max_gen_len           = 27
        last_metric           = ""
        FILTER                = 1
        path_results          = ""
        ft                    = 1
        instruct_yes          = 0
        apply_semantic_filter = args.semantic_filter
        similarity_threshold  = 0.7
        plot_distributions    = False

    ea = EvalArgs()
    test_ans = read_answers(ea.test_ans_file)
    model    = load_model(ea)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tests    = read_test_file(ea.input_file)
    counters = read_last_metric(ea.last_metric, ea.apply_semantic_filter)

    is_llama3 = "Llama-3" in model_name
    if is_llama3:
        patterns = [
            re.compile(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)', re.DOTALL),
            re.compile(r'\b(?:\d+[.:])?[\s_]*([\w\-()\/&,\.\'\s]+?)(?:[\]\[<]|$|,)'),
            re.compile(r'[^a-zA-Z0-9()\-_&\/\'\s]'),
        ]
        stop_chars = [torch.tensor([60], device="cuda:0"), torch.tensor([8], device="cuda:0")]
    else:
        patterns   = [re.compile(r'.*?[\d:@][._](.*?)[\]\[]?([< ].*?)?$')]
        stop_chars = [torch.tensor([29962], device="cuda:0"),
                      torch.tensor([29961], device="cuda:0"),
                      torch.tensor([4638],  device="cuda:0"),
                      torch.tensor([29871], device="cuda:0")]

    evaler = Evaler(topk=10, tests=tests, test_ans=test_ans,
                    eval_txt_path=metric_file, args=ea,
                    model=model, tokenizer=tokenizer,
                    patterns=patterns, early_stop_chars=stop_chars,
                    obligations=[])

    evaler.eval(counters, ea.begin, filter_yes=bool(ea.FILTER))
    print(f"\n[eval] results saved → {metric_file}")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = get_cfg(args.dataset)

    mode = args.mode
    print(f"\nRECIPE-TKG v2  |  dataset={args.dataset}  mode={mode}")

    align_path = full_path = checkpoint_dir = None

    run_preprocess = mode in ("all", "preprocess")
    run_retrieve   = mode in ("all", "retrieve")
    run_samples    = mode in ("all", "samples")
    run_train      = mode in ("all", "train")
    run_eval       = mode in ("all", "eval")

    if run_preprocess:
        stage_preprocess(args, cfg)

    if run_retrieve:
        stage_retrieve(args, cfg)

    if run_samples:
        align_path, full_path = stage_samples(args, cfg)

    if run_train:
        checkpoint_dir = stage_train(args, cfg, align_path, full_path)

    if run_eval:
        stage_eval(args, cfg, checkpoint_dir)

    print(f"\n[done] mode={mode} dataset={args.dataset}")


if __name__ == "__main__":
    main()
