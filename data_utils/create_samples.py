"""Build training JSON samples from history_facts.txt files.

Creates data/<DATASET>/processed/train_samples/
    <dataset>_rbmh_1024_align.json     — 1024 balanced samples for fine-tuning
    <dataset>_rbmh_full.json           — all samples (used as FULL_DATA_PATH)
"""

import json
import os
import random
from pathlib import Path


def _parse_history_facts(path: str) -> list:
    """Read history_facts.txt → list of sample strings."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    samples = [s.strip() for s in content.split("\n\n") if s.strip()]
    return samples


def _parse_answers(path: str) -> list:
    """Read test_ans.txt → list of answer strings (entity name)."""
    answers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                answers.append(parts[2] if len(parts) > 2 else parts[0])
    return answers


def _build_instruction(dataset: str) -> str:
    return (
        "<s>[INST] <<SYS>> "
        "You must be able to correctly predict the next {object_label} from "
        'a given text consisting of multiple quadruplets in the form of '
        '"{time}:[{subject}, {relation}, {object_label}.{object}]" '
        'and the query in the form of "{time}:[{subject}, {relation}," in the end.\n'
        "You must generate {object_label}.{object}\n\n<</SYS>>"
    )


def build_json_samples(dataset: str, data_root: str = "data",
                       retrieve_type: str = "rbmh",
                       n_align: int = 1024, seed: int = 42) -> tuple:
    """Build aligned + full JSON training files. Returns (align_path, full_path)."""
    proc_dir   = Path(data_root) / dataset / "processed"
    hist_file  = proc_dir / f"history_facts_{dataset.lower()}_{retrieve_type}.txt"
    ans_file   = proc_dir / f"test_ans_{dataset.lower()}.txt"

    if not hist_file.exists():
        raise FileNotFoundError(f"Run retrieval first: {hist_file}")
    if not ans_file.exists():
        raise FileNotFoundError(f"Missing answers file: {ans_file}")

    histories = _parse_history_facts(str(hist_file))
    answers   = _parse_answers(str(ans_file))

    assert len(histories) == len(answers), \
        f"Mismatch: {len(histories)} histories vs {len(answers)} answers"

    ins = _build_instruction(dataset)

    def _make_sample(hist, ans):
        # last line of hist is the query
        lines = hist.strip().split("\n")
        query = lines[-1]
        history_text = "\n".join(lines)
        context = ins + history_text + "[/INST]"
        # answer format: "id.name" — we use entity id from the query prefix
        # simplified: just use answer as target
        return {"context": context, "target": ans}

    samples = [_make_sample(h, a) for h, a in zip(histories, answers)]

    out_dir = proc_dir / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full set
    full_path = out_dir / f"{dataset.lower()}_{retrieve_type}_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # Aligned subset (balanced, max n_align)
    random.seed(seed)
    align = random.sample(samples, min(n_align, len(samples)))
    align_path = out_dir / f"{dataset.lower()}_{retrieve_type}_{n_align}_align.json"
    with open(align_path, "w", encoding="utf-8") as f:
        json.dump(align, f, ensure_ascii=False, indent=2)

    print(f"[samples] {dataset}: {len(samples)} total, {len(align)} aligned")
    print(f"[samples] saved → {align_path}")
    return str(align_path), str(full_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--retrieve_type", "-t", default="rbmh")
    p.add_argument("--n_align", type=int, default=1024)
    args = p.parse_args()
    build_json_samples(args.dataset, args.data_root, args.retrieve_type, args.n_align)
