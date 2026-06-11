"""RBMH / TLR / ISI history retriever.

Reads from  data/<DATASET>/processed/
Writes to   data/<DATASET>/processed/history_facts_<ds>_<type>.txt
                           processed/test_ans_<ds>.txt
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.basic import read_txt_as_list, write_txt_ans
from data_utils.weighted_sampling import Retriever
from dataset_config import get_cfg


# ---------------------------------------------------------------------------
# JSON helpers (avoids circular import with basic.py)
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_word_triples(path):
    """Read <split>_words.txt → list of tab-separated lines."""
    with open(path, encoding="utf-8") as f:
        return [l for l in f.readlines() if l.strip()]


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(dataset: str, split: str = "test",
             retrieve_type: str = "rbmh",
             rules_file: str = "",
             gamma_override: list = None,
             data_root: str = "data",
             n_jobs: int = 50) -> tuple:
    """Run retrieval for one split. Returns (idx_list, text_list)."""
    proc = Path(data_root) / dataset / "processed"
    cfg  = get_cfg(dataset)

    entities  = _load_json(proc / "entity2id.json")
    relations = _load_json(proc / "relation2id.json")
    times_id  = _load_json(proc / "ts2id.json")

    # Load split in word format
    word_file = proc / f"{split}_words.txt"
    if not word_file.exists():
        raise FileNotFoundError(f"Run preprocess first: {word_file}")
    data = _load_word_triples(str(word_file))

    # all_facts in word format
    all_facts_path = proc / "all_facts.txt"
    with open(all_facts_path, encoding="utf-8") as f:
        all_facts = f.readlines()

    # Rules
    rules = None
    rules_dir = Path(__file__).parent / "rules_output" / dataset.lower()
    if rules_file:
        rp = rules_dir / rules_file
        if rp.exists():
            rules = _load_json(str(rp))
        else:
            print(f"[WARN] rules file not found: {rp}")
    if rules is None:
        existing = sorted(glob.glob(str(rules_dir / "*rules.json")))
        if existing:
            rules = _load_json(existing[-1])
            print(f"[retrieve] auto-loaded rules: {existing[-1]}")

    gamma = gamma_override if gamma_override else cfg["gamma"]
    num_relations = cfg["num_relations"] or len(relations)
    rel_keys = list(relations.keys())

    print(f"[retrieve] {dataset}/{split}  type={retrieve_type}  gamma={gamma}")

    rtr = Retriever(
        gamma=gamma,
        test=data,
        all_facts=all_facts,
        entities=entities,
        relations=relations,
        times_id=times_id,
        num_relations=num_relations,
        chains=rules,
        rel_keys=rel_keys,
        dataset=dataset.lower(),
        retrieve_type=retrieve_type,
    )
    idx_list, text_list = rtr.get_output()
    return idx_list, text_list, data


def retrieve_and_save(dataset: str, split: str = "test",
                      retrieve_type: str = "rbmh",
                      rules_file: str = "",
                      gamma_override: list = None,
                      data_root: str = "data"):
    """Run retrieval and save outputs to processed/."""
    proc = Path(data_root) / dataset / "processed"

    idx_list, text_list, data = retrieve(
        dataset=dataset, split=split,
        retrieve_type=retrieve_type,
        rules_file=rules_file,
        gamma_override=gamma_override,
        data_root=data_root)

    ds_lower = dataset.lower()
    # history facts
    hist_path = proc / f"history_facts_{ds_lower}_{retrieve_type}.txt"
    with open(hist_path, "w", encoding="utf-8") as f:
        for entry in text_list:
            f.write(entry[0] + "\n")
    print(f"[retrieve] saved → {hist_path}")

    # test answers  (sub\trel\tobj\ttime format)
    ans_path = proc / f"test_ans_{ds_lower}.txt"
    write_txt_ans(str(ans_path), data, head="")
    print(f"[retrieve] saved → {ans_path}")
    return str(hist_path), str(ans_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--split", "-s", default="test",
                   choices=["train", "valid", "test"])
    p.add_argument("--retrieve_type", "-t", default="rbmh",
                   choices=["rbmh", "tlr", "isi"])
    p.add_argument("--rules_file", "-r", default="")
    p.add_argument("--data_root", default="data")
    p.add_argument("--n_jobs", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retrieve_and_save(
        dataset=args.dataset,
        split=args.split,
        retrieve_type=args.retrieve_type,
        rules_file=args.rules_file,
        data_root=args.data_root)
