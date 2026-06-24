"""
Data loading, graph indexing, and PyTorch Dataset for TREA-TKG.

Quadruple format (whitespace-separated):
    sub_id  rel_id  obj_id  timestamp  [0]

Supports ICEWS18 (step=24), YAGO (step=1), WIKI (step=1).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from trea.config import TREAConfig


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_quadruples(path: str) -> np.ndarray:
    """Return (N, 4) int32 array  [sub, rel, obj, time]."""
    quads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                quads.append([int(parts[0]), int(parts[1]),
                               int(parts[2]), int(parts[3])])
    return np.array(quads, dtype=np.int32)


def get_dataset_step(dataset: str) -> int:
    return 24 if dataset in ("ICEWS18", "ICEWS14") else 1


# ─────────────────────────────────────────────────────────────────────────────
# Graph Index  (built once, shared across train / valid / test)
# ─────────────────────────────────────────────────────────────────────────────

class GraphIndex:
    """
    Pre-built look-up structures over ALL known quadruples:

    a) all_answers[(s, r, t)]  → set of true objects   (for filtered eval)
    b) _by_time_sub[(t, s)]    → list[(rel, obj, time)] (history retrieval)
    c) _sro_times[(s, r, o)]   → sorted list of t       (copy score)
    """

    def __init__(self, quads_all: np.ndarray, step: int = 1):
        self.step = step

        # filtered answers
        self.all_answers: Dict[Tuple[int, int, int], set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        # history index: (time, subject) → facts
        self._by_time_sub: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = \
            defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o), int(t)))

        # copy counter: (s, r, o) → sorted timestamps
        self._sro_times: Dict[Tuple[int, int, int], List[int]] = \
            defaultdict(list)
        for s, r, o, t in quads_all:
            self._sro_times[(int(s), int(r), int(o))].append(int(t))
        for key in self._sro_times:
            self._sro_times[key].sort()

    # ── history retrieval ─────────────────────────────────────────────────────

    def get_history(self, sub: int, query_time: int,
                    history_len: int) -> List[Tuple[int, int, int]]:
        """
        Return facts for subject=sub in window [query_time - history_len*step, query_time).
        Sorted newest-first.  Caps at history_len × 10 facts.
        """
        facts: List[Tuple[int, int, int]] = []
        t = query_time - self.step
        steps_back = 0
        while steps_back < history_len and t >= 0:
            facts.extend(self._by_time_sub.get((t, sub), []))
            t -= self.step
            steps_back += 1
        facts.sort(key=lambda x: -x[2])
        return facts[:history_len * 10]

    # ── copy score vector ─────────────────────────────────────────────────────

    def get_copy_counts(self, sub: int, rel: int, query_time: int,
                        num_entities: int, copy_lambda: float,
                        step: int) -> np.ndarray:
        """
        copy[o] = log(1 + count(s,r,o, <t))
                × exp(−λ × (query_time − last_seen) / step)

        Novel: rare (s,r,o) patterns receive relative boost via log scaling.
        """
        scores = np.zeros(num_entities, dtype=np.float32)
        for (s_, r_, o_), times in self._sro_times.items():
            if s_ != sub or r_ != rel:
                continue
            count = sum(1 for tt in times if tt < query_time)
            if count == 0:
                continue
            last_t = max(tt for tt in times if tt < query_time)
            recency = np.exp(-copy_lambda * (query_time - last_t) / max(step, 1))
            if o_ < num_entities:
                scores[o_] = np.log1p(count) * recency
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataset(Dataset):
    def __init__(self, quads: np.ndarray, use_inverse: bool = False,
                 num_relations: Optional[int] = None):
        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:, 2],
                            quads[:, 1] + num_relations,
                            quads[:, 0],
                            quads[:, 3]], axis=1).astype(np.int32)
            self.data = np.concatenate([quads, inv], axis=0)
        else:
            self.data = quads.copy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        s, r, o, t = self.data[idx]
        return int(s), int(r), int(o), int(t)


# ─────────────────────────────────────────────────────────────────────────────
# High-level DataLoader wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataLoader:
    """Loads all splits, builds GraphIndex, exposes Dataset objects."""

    def __init__(self, cfg: TREAConfig):
        self.cfg = cfg
        base = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_dataset_step(cfg.dataset)

        train_q = load_quadruples(os.path.join(base, "train.txt"))
        valid_q = load_quadruples(os.path.join(base, "valid.txt"))
        test_q  = load_quadruples(os.path.join(base, "test.txt"))

        all_q = np.concatenate([train_q, valid_q, test_q], axis=0)
        self.num_entities  = int(all_q[:, [0, 2]].max()) + 1
        self.num_relations = int(all_q[:, 1].max()) + 1

        self.index = GraphIndex(all_q, step=self.step)

        self.train_set = TKGDataset(train_q, use_inverse=cfg.use_inverse,
                                    num_relations=self.num_relations)
        self.valid_set = TKGDataset(valid_q, use_inverse=False)
        self.test_set  = TKGDataset(test_q,  use_inverse=False)

        print(
            f"[{cfg.dataset}] entities={self.num_entities:,}  "
            f"relations={self.num_relations}  "
            f"train={len(self.train_set):,}  "
            f"valid={len(self.valid_set):,}  "
            f"test={len(self.test_set):,}  "
            f"step={self.step}"
        )

    # ── batch builders ────────────────────────────────────────────────────────

    def build_history_batch(
        self,
        subs: List[int], rels: List[int], times: List[int], max_hist: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Returns padded tensors:
            hist_rels  (B, L)   – relation ids
            hist_objs  (B, L)   – object ids
            hist_times (B, L)   – raw timestamps
            hist_mask  (B, L)   – True = valid position
        """
        B = len(subs)
        L = max_hist * 10  # max facts per query

        h_rels  = torch.zeros(B, L, dtype=torch.long)
        h_objs  = torch.zeros(B, L, dtype=torch.long)
        h_times = torch.zeros(B, L, dtype=torch.long)
        h_mask  = torch.zeros(B, L, dtype=torch.bool)

        for i, (s, r, t) in enumerate(zip(subs, rels, times)):
            facts = self.index.get_history(s, t, max_hist)
            for j, (fr, fo, ft) in enumerate(facts[:L]):
                h_rels[i, j]  = fr
                h_objs[i, j]  = fo
                h_times[i, j] = ft
                h_mask[i, j]  = True

        return h_rels, h_objs, h_times, h_mask

    def get_copy_scores_batch(
        self, subs: List[int], rels: List[int], times: List[int],
    ) -> torch.Tensor:
        """Return (B, num_entities) float32 copy score matrix."""
        B = len(subs)
        scores = torch.zeros(B, self.num_entities, dtype=torch.float32)
        for i, (s, r, t) in enumerate(zip(subs, rels, times)):
            sc = self.index.get_copy_counts(
                s, r, t, self.num_entities, self.cfg.copy_lambda, self.step
            )
            scores[i] = torch.from_numpy(sc)
        return scores
