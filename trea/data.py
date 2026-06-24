"""
Data loading and graph indexing for AURORA-TKG.

Quadruple format (whitespace-separated):
    sub_id  rel_id  obj_id  timestamp  [0]

New additions vs TREA-TKG:
  - build_neighborhood_batch(): R-GAT neighborhood tensors
  - entity copy scores: (s, any_r, o, <t) co-occurrence signal
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from trea.config import AURORAConfig


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
# Graph Index
# ─────────────────────────────────────────────────────────────────────────────

class GraphIndex:
    """
    Pre-built look-up structures:

    a) all_answers[(s, r, t)]    → set of true objects (filtered eval)
    b) _by_time_sub[(t, s)]      → list[(rel, obj, time)] (neighborhood)
    c) _sro_times[(s, r, o)]     → sorted list of t (relation copy)
    d) _by_sub[s]                → list[(o, last_t, count)] (entity copy)
    """

    def __init__(self, quads_all: np.ndarray, step: int = 1):
        self.step = step

        # filtered answers
        self.all_answers: Dict[Tuple[int, int, int], set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        # history / neighborhood index: (time, subject) → facts
        self._by_time_sub: Dict[Tuple[int, int], List[Tuple[int, int]]] = \
            defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o)))

        # relation copy: (s, r) → {o: sorted_timestamps}
        # O(1) lookup instead of O(all_triples) scan
        _sro_raw: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for s, r, o, t in quads_all:
            _sro_raw[(int(s), int(r), int(o))].append(int(t))
        for key in _sro_raw:
            _sro_raw[key].sort()

        self._sr_objs: Dict[Tuple[int, int], Dict[int, List[int]]] = \
            defaultdict(dict)
        for (s, r, o), times in _sro_raw.items():
            self._sr_objs[(s, r)][o] = times

        # entity copy: s → {o: (last_t_before_any_t, total_count_before_any_t)}
        # We store the raw list and compute at query time for correct filtering.
        self._by_sub: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_sub[int(s)].append((int(o), int(t), int(r)))

    # ── relation copy score vector ─────────────────────────────────────────────

    def get_rel_copy_scores(self, sub: int, rel: int, query_time: int,
                            num_entities: int, copy_lambda: float,
                            recency_steps: int = 3,
                            recency_boost: float = 5.0) -> np.ndarray:
        """
        copy[o] = log(1+count) × exp(−λ × (t − last_t) / step)
        O(|neighbors of (s,r)|) — not O(all triples).
        Recency burst: × recency_boost if within last recency_steps snapshots.
        """
        scores = np.zeros(num_entities, dtype=np.float32)
        step   = max(self.step, 1)
        recency_threshold = recency_steps * step

        for o, times in self._sr_objs.get((sub, rel), {}).items():
            ts = [tt for tt in times if tt < query_time]
            if not ts:
                continue
            count  = len(ts)
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            score  = np.log1p(count) * decay
            if (query_time - last_t) <= recency_threshold:
                score *= recency_boost
            if o < num_entities:
                scores[o] = score
        return scores

    # ── entity copy score vector ───────────────────────────────────────────────

    def get_ent_copy_scores(self, sub: int, query_time: int,
                            num_entities: int, copy_lambda: float) -> np.ndarray:
        """
        ent_copy[o] = log(1+count(s,*,o,<t)) × exp(−λ × Δt_last / step)
        Any relation between s and o counts.
        """
        scores  = np.zeros(num_entities, dtype=np.float32)
        step    = max(self.step, 1)
        by_obj: Dict[int, List[int]] = defaultdict(list)
        for o, t, _ in self._by_sub.get(sub, []):
            if t < query_time:
                by_obj[o].append(t)
        for o, ts in by_obj.items():
            if o >= num_entities:
                continue
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            scores[o] = np.log1p(len(ts)) * decay
        return scores

    # ── neighborhood for R-GAT ────────────────────────────────────────────────

    def get_snapshot_neighbors(
        self, sub: int, query_time: int, long_len: int, k_neighbors: int,
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For entity `sub`, build neighbor tensors over last `long_len` snapshots.

        Returns:
            neigh_ents  (long_len, k_neighbors) int32 — entity IDs (0 = pad)
            neigh_rels  (long_len, k_neighbors) int32 — relation IDs
            neigh_mask  (long_len, k_neighbors) bool
        """
        H, K = long_len, k_neighbors
        neigh_ents = np.zeros((H, K), dtype=np.int32)
        neigh_rels = np.zeros((H, K), dtype=np.int32)
        neigh_mask = np.zeros((H, K), dtype=bool)

        t = query_time - self.step
        for h in range(H):
            if t < 0:
                break
            facts = self._by_time_sub.get((t, sub), [])
            if facts:
                if len(facts) > K:
                    chosen = rng.sample(facts, K)
                else:
                    chosen = facts
                for k, (r, o) in enumerate(chosen):
                    neigh_ents[h, k] = o
                    neigh_rels[h, k] = r
                    neigh_mask[h, k] = True
            t -= self.step

        return neigh_ents, neigh_rels, neigh_mask


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

    def __init__(self, cfg: AURORAConfig):
        self.cfg  = cfg
        base      = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_dataset_step(cfg.dataset)
        self._rng = random.Random(cfg.seed)

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

    # ── R-GAT neighborhood batch ──────────────────────────────────────────────

    def build_neighborhood_batch(
        self,
        subs: List[int],
        times: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build padded neighbor tensors for a batch of (sub, time) pairs.

        Returns:
            neigh_ents  (B, H, K)  long  — entity IDs
            neigh_rels  (B, H, K)  long  — relation IDs
            neigh_mask  (B, H, K)  bool  — validity mask
        """
        cfg = self.cfg
        B   = len(subs)
        H   = cfg.long_len
        K   = cfg.k_neighbors

        ne = np.zeros((B, H, K), dtype=np.int32)
        nr = np.zeros((B, H, K), dtype=np.int32)
        nm = np.zeros((B, H, K), dtype=bool)

        for i, (s, t) in enumerate(zip(subs, times)):
            ne[i], nr[i], nm[i] = self.index.get_snapshot_neighbors(
                s, t, H, K, self._rng
            )

        return (torch.from_numpy(ne).long(),
                torch.from_numpy(nr).long(),
                torch.from_numpy(nm))

    # ── copy score batch ──────────────────────────────────────────────────────

    def get_rel_copy_batch(
        self, subs: List[int], rels: List[int], times: List[int],
    ) -> torch.Tensor:
        """Return (B, num_entities) float32 relation-copy score matrix."""
        B = len(subs)
        out = torch.zeros(B, self.num_entities, dtype=torch.float32)
        for i, (s, r, t) in enumerate(zip(subs, rels, times)):
            sc = self.index.get_rel_copy_scores(
                s, r, t, self.num_entities,
                self.cfg.copy_lambda,
                self.cfg.recency_steps,
                self.cfg.recency_boost,
            )
            out[i] = torch.from_numpy(sc)
        return out

    def get_ent_copy_batch(
        self, subs: List[int], times: List[int],
    ) -> torch.Tensor:
        """Return (B, num_entities) float32 entity-copy score matrix."""
        B = len(subs)
        out = torch.zeros(B, self.num_entities, dtype=torch.float32)
        if not self.cfg.use_entity_copy:
            return out
        for i, (s, t) in enumerate(zip(subs, times)):
            sc = self.index.get_ent_copy_scores(
                s, t, self.num_entities, self.cfg.copy_lambda
            )
            out[i] = torch.from_numpy(sc)
        return out
