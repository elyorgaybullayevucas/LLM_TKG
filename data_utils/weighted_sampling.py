"""RBMH weighted retriever.

Expects word-format triples:
    sub_name\\trel_name\\tobj_name\\ttime_str\\t0

Weight formula per candidate fact i:
    w(i) = neighbor_w × frequency_w × (time_w + connection_w + vip_w)
"""

from collections import defaultdict
from math import exp, log

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class Retriever:
    def __init__(self, gamma, test, all_facts, entities, relations, times_id,
                 num_relations, chains, rel_keys, dataset, retrieve_type="rbmh"):
        self.gamma = gamma
        self.retrieve_type = retrieve_type
        self.dataset = dataset.lower()
        self.test = test          # list of word-format lines
        self.all_facts = all_facts
        self.entities = entities            # {name: id}
        self.relations = relations          # {name: id}
        self.times_id = times_id            # {str(t): t}
        self.num_relations = num_relations
        self.chains = chains or {}
        self.rel_keys = np.array(rel_keys)  # index→rel_name

        # Parse columns once
        def _col(idx):
            return np.array([r.strip().split("\t")[idx] for r in all_facts
                             if r.strip() and len(r.strip().split("\t")) > idx])

        self.col_sub  = _col(0)
        self.col_rel  = _col(1)
        self.col_obj  = _col(2)
        self.col_time = _col(3)   # string time IDs

        # Numeric time for fast comparison
        self.col_time_num = np.array([int(t) for t in self.col_time])

    # ------------------------------------------------------------------
    # Lazy indexes (rebuilt per-query when needed)
    # ------------------------------------------------------------------

    def _build_time_index(self, cur_t: int):
        self.time_index = defaultdict(int)
        for s, o, t in zip(self.col_sub, self.col_obj, self.col_time_num):
            if t < cur_t:
                self.time_index[(s, o)] += 1

    def _build_graph(self, cur_t: int):
        self.graph = nx.Graph()
        edges = [(s, o) for s, o, t in
                 zip(self.col_sub, self.col_obj, self.col_time_num) if t <= cur_t]
        self.graph.add_edges_from(edges)

    def _build_freq(self, cur_t: int):
        keys = list(zip(
            self.col_sub[self.col_time_num < cur_t],
            self.col_rel[self.col_time_num < cur_t],
            self.col_obj[self.col_time_num < cur_t]))
        if not keys:
            self.triple_count = {}
            return
        unique, counts = np.unique(keys, return_counts=True, axis=0)
        self.triple_count = {tuple(k): int(v) for k, v in zip(unique, counts)}

    # ------------------------------------------------------------------
    # Weight components
    # ------------------------------------------------------------------

    def _nw(self, hs, ho):
        return exp(-self.gamma[0] * (hs + ho - 1))

    def _fw(self, i):
        key = (self.col_sub[i], self.col_rel[i], self.col_obj[i])
        ns = self.triple_count.get(key, 1)
        return 1.0 / (self.gamma[1] * log(max(ns, 1)) + 1)

    def _tw(self, t_num, cur_t):
        return exp(-self.gamma[2] * (cur_t - t_num))

    def _cw(self, s, o):
        c = self.time_index.get((s, o), 0) + self.time_index.get((o, s), 0)
        val = log(1 + self.gamma[3] * c)
        return val / (1 + val)

    def _total_w(self, i, s, o, t_num, cur_t, hops, vip):
        hs = hops.get(s, np.inf)
        ho = hops.get(o, np.inf)
        if np.isinf(hs) or np.isinf(ho):
            return 0.0
        vip_w = 1 if (s in vip or o in vip) else 0
        return self._nw(hs, ho) * self._fw(i) * (self._tw(t_num, cur_t) + self._cw(s, o) + vip_w)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _top_sample(self, weights, k, seed=42, mult=10):
        weights = np.array(weights)
        total = weights.sum()
        if total == 0:
            return []
        np.random.seed(seed)
        top_k = min(len(weights), mult * k)
        top_idx = np.argpartition(-weights, top_k - 1)[:top_k]
        probs = weights[top_idx] / weights[top_idx].sum()
        actual = min(k, int(np.count_nonzero(probs)))
        if actual == 0:
            return []
        chosen = np.random.choice(len(top_idx), size=actual, replace=False, p=probs)
        return top_idx[chosen].tolist()

    # ------------------------------------------------------------------
    # Query preparation
    # ------------------------------------------------------------------

    def _prepare(self, i):
        parts = self.test[i].strip().split("\t")
        if len(parts) < 4:
            return None
        sub, rel, _, time_str = parts[0], parts[1], parts[2], parts[3]
        cur_t = int(time_str)
        # index of this test triple in all_facts (last block)
        idx_test = len(self.all_facts) - len(self.test) + i
        mask_t = self.col_time_num < cur_t
        mask_s = self.col_sub == sub
        s0 = set(np.where(mask_t & mask_s)[0])
        s0.discard(idx_test)
        head_rel = self.relations.get(rel, -1)
        return s0, cur_t, head_rel, sub, rel

    # ------------------------------------------------------------------
    # History formatting
    # ------------------------------------------------------------------

    def _period(self):
        return 24 if self.dataset in ("icews14", "icews18") else 1

    def _collect_hist(self, facts, n):
        period = self._period()
        out = []
        for fact in reversed(facts[:n]):
            parts = fact.strip().split("\t")
            if len(parts) < 4:
                continue
            s, r, o, t = parts[0], parts[1], parts[2], parts[3]
            t_num = int(t)
            oid = self.entities.get(o, 0)
            out.append(f"{t_num // period}: [{s}, {r}, {oid}.{o}] \n")
        return out

    def _build_query(self, cur_t, sub, rel, hist=""):
        period = self._period()
        return ["".join(hist) + f"{cur_t // period}: [{sub}, {rel},\n"]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _weighted(self, i):
        n = 50
        prep = self._prepare(i)
        if prep is None:
            return [], [""]
        s0, cur_t, head_rel, sub, rel = prep

        chain_key = str(head_rel)
        if chain_key not in self.chains:
            return [], self._build_query(cur_t, sub, rel)

        s0_arr = np.array(list(s0))
        idx = []
        for chain in self.chains[chain_key]:
            if len(chain["body_rels"]) != 1:
                continue
            body_rel = chain["body_rels"][-1] % self.num_relations
            r_name = self.rel_keys[body_rel]
            mask = np.where(self.col_rel == r_name)[0]
            hits = np.intersect1d(mask, s0_arr).tolist()
            idx = list(set(idx + hits))
            if len(idx) >= n:
                break

        self._build_freq(cur_t)
        if len(idx) < n:
            vip = set(self.col_obj[np.array(idx)].tolist()) if idx else set()
            self._build_time_index(cur_t)
            self._build_graph(cur_t)
            try:
                hops = nx.single_source_shortest_path_length(self.graph, sub)
            except Exception:
                hops = {}
            if hops:
                ws = []
                for j in range(len(self.col_sub)):
                    t_j = self.col_time_num[j]
                    if t_j < cur_t and self.col_sub[j] != sub:
                        ws.append(self._total_w(j, self.col_sub[j], self.col_obj[j],
                                                t_j, cur_t, hops, vip))
                    else:
                        ws.append(0.0)
                extra = self._top_sample(ws, n - len(idx))
                idx = list(set(idx) | set(extra))

        idx = sorted(idx, reverse=True)[:n]
        facts = [self.all_facts[j] for j in idx]
        if not facts:
            return idx, self._build_query(cur_t, sub, rel)
        hist = self._collect_hist(facts, len(facts))
        return idx, self._build_query(cur_t, sub, rel, hist)

    def _tlr(self, i):
        n = 50
        prep = self._prepare(i)
        if prep is None:
            return [], [""]
        s0, cur_t, head_rel, sub, rel = prep

        chain_key = str(head_rel)
        if chain_key not in self.chains:
            return [], self._build_query(cur_t, sub, rel)

        s0_arr = np.array(list(s0))
        idx = []
        for chain in self.chains[chain_key]:
            if len(chain["body_rels"]) != 1:
                continue
            body_rel = chain["body_rels"][-1] % self.num_relations
            mask = np.where(self.col_rel == self.rel_keys[body_rel])[0]
            hits = np.intersect1d(mask, s0_arr).tolist()
            idx = list(set(idx + hits))
            if len(idx) >= n:
                break

        idx = sorted(idx, reverse=True)[:n]
        facts = [self.all_facts[j] for j in idx]
        if not facts:
            return idx, self._build_query(cur_t, sub, rel)
        return idx, self._build_query(cur_t, sub, rel, self._collect_hist(facts, len(facts)))

    def _isi(self, i):
        n = 50
        prep = self._prepare(i)
        if prep is None:
            return [], [""]
        s0, cur_t, _, sub, rel = prep
        idx = sorted(list(s0), reverse=True)[:n]
        facts = [self.all_facts[j] for j in idx]
        if not facts:
            return idx, self._build_query(cur_t, sub, rel)
        return idx, self._build_query(cur_t, sub, rel, self._collect_hist(facts, len(facts)))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_output(self, n_jobs=50):
        fn = {"rbmh": self._weighted, "tlr": self._tlr, "isi": self._isi}.get(
            self.retrieve_type, self._weighted)
        results = Parallel(n_jobs=n_jobs)(
            delayed(fn)(i) for i in tqdm(range(len(self.test)),
                                         desc=f"[{self.dataset}] {self.retrieve_type}"))
        idx_list, text_list = zip(*results)
        return list(idx_list), list(text_list)
