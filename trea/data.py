"""
Data loading for AURORA-TKG.

KEY DESIGN: All neighborhoods and copy scores are pre-computed ONCE at
startup and stored as numpy arrays. __getitem__ is O(1) — no CPU work
during training, GPU stays busy.

Quadruple format: sub_id rel_id obj_id timestamp [0]
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
# Graph Index  (build-once lookup structures)
# ─────────────────────────────────────────────────────────────────────────────

class GraphIndex:
    def __init__(self, quads_all: np.ndarray, step: int = 1):
        self.step = step

        # filtered answers
        self.all_answers: Dict[Tuple[int,int,int], set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        # neighborhood: (time, sub) → list[(rel, obj)]
        self._by_time_sub: Dict[Tuple[int,int], List[Tuple[int,int]]] = \
            defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_time_sub[(int(t), int(s))].append((int(r), int(o)))

        # relation copy: (s,r) → {o: sorted_timestamps}  — O(1) lookup
        _sr_raw: Dict[Tuple[int,int,int], List[int]] = defaultdict(list)
        for s, r, o, t in quads_all:
            _sr_raw[(int(s), int(r), int(o))].append(int(t))
        for key in _sr_raw:
            _sr_raw[key].sort()
        self._sr_objs: Dict[Tuple[int,int], Dict[int,List[int]]] = \
            defaultdict(dict)
        for (s, r, o), times in _sr_raw.items():
            self._sr_objs[(s, r)][o] = times

        # entity copy: sub → list[(obj, time, rel)]
        self._by_sub: Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)
        for s, r, o, t in quads_all:
            self._by_sub[int(s)].append((int(o), int(t), int(r)))

    def get_rel_copy_scores(self, sub, rel, query_time, num_entities,
                            copy_lambda, recency_steps=2, recency_boost=4.0):
        scores = np.zeros(num_entities, dtype=np.float32)
        step   = max(self.step, 1)
        thr    = recency_steps * step
        for o, times in self._sr_objs.get((sub, rel), {}).items():
            ts = [tt for tt in times if tt < query_time]
            if not ts:
                continue
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            score  = np.log1p(len(ts)) * decay
            if (query_time - last_t) <= thr:
                score *= recency_boost
            if o < num_entities:
                scores[o] = score
        return scores

    def get_ent_copy_scores(self, sub, query_time, num_entities,
                            copy_lambda, recency_steps=2, recency_boost=4.0):
        scores  = np.zeros(num_entities, dtype=np.float32)
        step    = max(self.step, 1)
        thr     = recency_steps * step
        by_obj: Dict[int, List[int]] = defaultdict(list)
        for o, t, _ in self._by_sub.get(sub, []):
            if t < query_time:
                by_obj[o].append(t)
        for o, ts in by_obj.items():
            if o >= num_entities:
                continue
            last_t = max(ts)
            decay  = np.exp(-copy_lambda * (query_time - last_t) / step)
            score  = np.log1p(len(ts)) * decay
            if (query_time - last_t) <= thr:
                score *= recency_boost
            scores[o] = score
        return scores

    def get_snapshot_neighbors(self, sub, query_time, long_len, k_neighbors, rng):
        H, K = long_len, k_neighbors
        ne = np.zeros((H, K), dtype=np.int32)
        nr = np.zeros((H, K), dtype=np.int32)
        nm = np.zeros((H, K), dtype=bool)
        t  = query_time - self.step
        for h in range(H):
            if t < 0:
                break
            facts = self._by_time_sub.get((t, sub), [])
            if facts:
                chosen = rng.sample(facts, min(len(facts), K))
                for k, (r, o) in enumerate(chosen):
                    ne[h, k] = o
                    nr[h, k] = r
                    nm[h, k] = True
            t -= self.step
        return ne, nr, nm


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed Dataset  (O(1) __getitem__)
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataset(Dataset):
    """
    All neighborhoods and copy scores pre-computed at init time.
    __getitem__ is a pure numpy index — zero CPU work during training.
    """
    def __init__(self, quads: np.ndarray, index: GraphIndex,
                 num_entities: int, cfg: AURORAConfig,
                 use_inverse: bool = False, num_relations: int = None,
                 precompute: bool = True):

        step = get_dataset_step(cfg.dataset)

        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:,2], quads[:,1]+num_relations,
                            quads[:,0], quads[:,3]], axis=1).astype(np.int32)
            data = np.concatenate([quads, inv], axis=0)
        else:
            data = quads.copy()
        self.data = data
        N = len(data)

        if not precompute:
            # lightweight mode for valid/test (pre-compute on-the-fly in eval)
            self._neigh_ents = None
            return

        H  = cfg.long_len
        K  = cfg.k_neighbors
        rng = random.Random(cfg.seed)

        print(f"  Pre-computing neighborhoods ({N:,} samples, H={H}, K={K}) …",
              flush=True)
        ne = np.zeros((N, H, K), dtype=np.int16)   # int16 saves RAM
        nr = np.zeros((N, H, K), dtype=np.int16)
        nm = np.zeros((N, H, K), dtype=bool)

        for i in range(N):
            s, _, _, t = data[i]
            ne_i, nr_i, nm_i = index.get_snapshot_neighbors(
                int(s), int(t), H, K, rng)
            ne[i] = ne_i.astype(np.int16)
            nr[i] = nr_i.astype(np.int16)
            nm[i] = nm_i
            if (i+1) % 100_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self._neigh_ents = ne
        self._neigh_rels = nr
        self._neigh_mask = nm

        print(f"  Pre-computing copy scores …", flush=True)
        # Store sparse: list of (indices_array, values_array) per sample
        rc_idx = [None] * N
        rc_val = [None] * N
        ec_idx = [None] * N
        ec_val = [None] * N

        for i in range(N):
            s, r, _, t = data[i]
            rc = index.get_rel_copy_scores(int(s), int(r), int(t),
                                           num_entities, cfg.copy_lambda,
                                           cfg.recency_steps, cfg.recency_boost)
            nz = rc.nonzero()[0]
            rc_idx[i] = nz.astype(np.int32)
            rc_val[i] = rc[nz]

            if cfg.use_entity_copy:
                ec = index.get_ent_copy_scores(int(s), int(t), num_entities,
                                               cfg.copy_lambda,
                                               cfg.recency_steps,
                                               cfg.recency_boost)
                nze = ec.nonzero()[0]
                ec_idx[i] = nze.astype(np.int32)
                ec_val[i] = ec[nze]
            else:
                ec_idx[i] = np.zeros(0, dtype=np.int32)
                ec_val[i] = np.zeros(0, dtype=np.float32)

            if (i+1) % 100_000 == 0:
                print(f"    {i+1:,}/{N:,}", flush=True)

        self._rc_idx  = rc_idx
        self._rc_val  = rc_val
        self._ec_idx  = ec_idx
        self._ec_val  = ec_val
        self._num_ent = num_entities
        print(f"  Pre-computation done.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, r, o, t = self.data[idx]
        if self._neigh_ents is None:
            return int(s), int(r), int(o), int(t), None, None, None, None, None
        ne = torch.from_numpy(self._neigh_ents[idx].astype(np.int32)).long()
        nr = torch.from_numpy(self._neigh_rels[idx].astype(np.int32)).long()
        nm = torch.from_numpy(self._neigh_mask[idx])
        # sparse copy → will be densified in collate_fn
        return (int(s), int(r), int(o), int(t),
                ne, nr, nm,
                self._rc_idx[idx], self._rc_val[idx],
                self._ec_idx[idx], self._ec_val[idx])


def aurora_collate(batch):
    """Custom collate: densifies sparse copy score arrays."""
    (subs, rels, objs, times,
     ne_list, nr_list, nm_list,
     rc_idx_list, rc_val_list,
     ec_idx_list, ec_val_list) = zip(*batch)

    subs  = torch.tensor(subs,  dtype=torch.long)
    rels  = torch.tensor(rels,  dtype=torch.long)
    objs  = torch.tensor(objs,  dtype=torch.long)
    times = torch.tensor(times, dtype=torch.long)
    ne    = torch.stack(ne_list)
    nr    = torch.stack(nr_list)
    nm    = torch.stack(nm_list)

    B = len(subs)
    # infer num_entities from max index in rc_idx
    all_idx = np.concatenate([x for x in rc_idx_list if len(x) > 0] +
                              [x for x in ec_idx_list if len(x) > 0])
    N = int(all_idx.max()) + 1 if len(all_idx) > 0 else 1

    rc = torch.zeros(B, N, dtype=torch.float32)
    ec = torch.zeros(B, N, dtype=torch.float32)
    for i in range(B):
        if len(rc_idx_list[i]) > 0:
            rc[i, rc_idx_list[i]] = torch.from_numpy(rc_val_list[i])
        if len(ec_idx_list[i]) > 0:
            ec[i, ec_idx_list[i]] = torch.from_numpy(ec_val_list[i])

    return subs, rels, objs, times, ne, nr, nm, rc, ec


# ─────────────────────────────────────────────────────────────────────────────
# High-level DataLoader wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataLoader:
    def __init__(self, cfg: AURORAConfig):
        self.cfg  = cfg
        base      = os.path.join(cfg.data_dir, cfg.dataset)
        self.step = get_dataset_step(cfg.dataset)

        train_q = load_quadruples(os.path.join(base, "train.txt"))
        valid_q = load_quadruples(os.path.join(base, "valid.txt"))
        test_q  = load_quadruples(os.path.join(base, "test.txt"))

        all_q = np.concatenate([train_q, valid_q, test_q], axis=0)
        self.num_entities  = int(all_q[:, [0,2]].max()) + 1
        self.num_relations = int(all_q[:, 1].max()) + 1

        print(f"[{cfg.dataset}] entities={self.num_entities:,}  "
              f"relations={self.num_relations}  "
              f"train={len(train_q):,}  valid={len(valid_q):,}  "
              f"test={len(test_q):,}  step={self.step}")

        self.index = GraphIndex(all_q, step=self.step)

        print("Building train dataset (pre-computing) …")
        self.train_set = TKGDataset(
            train_q, self.index, self.num_entities, cfg,
            use_inverse=cfg.use_inverse,
            num_relations=self.num_relations,
            precompute=True,
        )
        print("Building valid dataset …")
        self.valid_set = TKGDataset(
            valid_q, self.index, self.num_entities, cfg,
            precompute=True,
        )
        print("Building test dataset …")
        self.test_set = TKGDataset(
            test_q, self.index, self.num_entities, cfg,
            precompute=True,
        )
        print("Datasets ready.")

    # kept for backward compatibility with evaluate.py
    def build_neighborhood_batch(self, subs, times):
        cfg = self.cfg
        B, H, K = len(subs), cfg.long_len, cfg.k_neighbors
        rng = random.Random(0)
        ne = np.zeros((B,H,K), dtype=np.int32)
        nr = np.zeros((B,H,K), dtype=np.int32)
        nm = np.zeros((B,H,K), dtype=bool)
        for i,(s,t) in enumerate(zip(subs,times)):
            ne[i],nr[i],nm[i] = self.index.get_snapshot_neighbors(s,t,H,K,rng)
        return (torch.from_numpy(ne).long(),
                torch.from_numpy(nr).long(),
                torch.from_numpy(nm))

    def get_rel_copy_batch(self, subs, rels, times):
        B = len(subs)
        out = torch.zeros(B, self.num_entities, dtype=torch.float32)
        for i,(s,r,t) in enumerate(zip(subs,rels,times)):
            sc = self.index.get_rel_copy_scores(
                s,r,t,self.num_entities,self.cfg.copy_lambda,
                self.cfg.recency_steps,self.cfg.recency_boost)
            out[i] = torch.from_numpy(sc)
        return out

    def get_ent_copy_batch(self, subs, times):
        B = len(subs)
        out = torch.zeros(B, self.num_entities, dtype=torch.float32)
        if not self.cfg.use_entity_copy:
            return out
        for i,(s,t) in enumerate(zip(subs,times)):
            sc = self.index.get_ent_copy_scores(
                s,t,self.num_entities,self.cfg.copy_lambda,
                self.cfg.recency_steps,self.cfg.recency_boost)
            out[i] = torch.from_numpy(sc)
        return out
