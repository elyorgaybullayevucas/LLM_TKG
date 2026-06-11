"""Preprocessing — converts raw dataset files into the format RECIPE-TKG expects.

Raw format  (data/<DATASET>/):
    entity2id.txt   — "<name>\\t<id>" (YAGO has 2 extra date columns)
    relation2id.txt — "<name>\\t<id>"
    train.txt       — "<sub_id> <rel_id> <obj_id> <time_id> 0"  (space/tab separated)
    valid.txt       — same
    test.txt        — same

Output  (data/<DATASET>/processed/):
    entity2id.json      {name: id}
    relation2id.json    {name: id}
    ts2id.json          {str(time_id): time_id}
    all_facts.txt       sub_name\\trel_name\\tobj_name\\ttime_str\\t0
    train_words.txt     same format, train split only
    valid_words.txt
    test_words.txt
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

from dataset_config import get_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_id_file(path: str, name_col: int = 0, id_col: int = 1) -> Dict[str, int]:
    """Read a tab-separated id file → {name: int_id}."""
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(name_col, id_col):
                continue
            try:
                d[parts[name_col].strip()] = int(parts[id_col].strip())
            except ValueError:
                pass
    return d


def _parse_triples(path: str) -> list:
    """Read a triples file → list of (sub_id, rel_id, obj_id, time_id)."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 4:
                continue
            try:
                rows.append(tuple(int(x) for x in parts[:4]))
            except ValueError:
                pass
    return rows


def _to_words(triples, id2ent, id2rel) -> list:
    """Convert (sub_id, rel_id, obj_id, time_id) → word-format lines."""
    lines = []
    for sub, rel, obj, t in triples:
        s = id2ent.get(sub, str(sub))
        r = id2rel.get(rel, str(rel))
        o = id2ent.get(obj, str(obj))
        lines.append(f"{s}\t{r}\t{o}\t{t}\t0")
    return lines


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(dataset: str, data_root: str = "data") -> str:
    """Build processed/ files for `dataset`. Returns path to processed dir."""
    raw_dir = Path(data_root) / dataset
    out_dir = raw_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. entity2id & relation2id ---
    ent2id = _parse_id_file(str(raw_dir / "entity2id.txt"))
    rel2id = _parse_id_file(str(raw_dir / "relation2id.txt"))

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    # --- 2. Collect all triples ---
    all_triples = []
    split_triples = {}
    for split in ("train", "valid", "test"):
        p = raw_dir / f"{split}.txt"
        rows = _parse_triples(str(p)) if p.exists() else []
        split_triples[split] = rows
        all_triples.extend(rows)

    # --- 3. ts2id: identity mapping str(t) → t ---
    all_times = sorted({t for *_, t in all_triples})
    ts2id = {str(t): t for t in all_times}

    # --- 4. Save JSONs ---
    def _save_json(obj, name):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    _save_json(ent2id, "entity2id.json")
    _save_json(rel2id, "relation2id.json")
    _save_json(ts2id,  "ts2id.json")

    # --- 5. Word-format split files ---
    for split, rows in split_triples.items():
        lines = _to_words(rows, id2ent, id2rel)
        with open(out_dir / f"{split}_words.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # --- 6. all_facts.txt = train + valid + test ---
    all_lines = _to_words(all_triples, id2ent, id2rel)
    with open(out_dir / "all_facts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"[preprocess] {dataset}: {len(ent2id)} entities, "
          f"{len(rel2id)} relations, {len(ts2id)} timestamps")
    print(f"[preprocess] saved to {out_dir}")
    return str(out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--data_root", default="data")
    args = p.parse_args()
    preprocess(args.dataset, args.data_root)
