"""
Filtered evaluation: MRR, Hits@1, Hits@3, Hits@10.

Standard TKG forecasting protocol:
  For each (s, r, o_true, t):
    1. Score all entities.
    2. Set scores of OTHER known-true answers to −∞ (filter).
    3. Rank o_true; accumulate MRR and Hits@k.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm

from trea.model import TREAModel
from trea.data import TKGDataLoader


@torch.no_grad()
def evaluate(
    model: TREAModel,
    loader: TKGDataLoader,
    split: str = "valid",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    max_hist: int = 10,
    hits_at: tuple = (1, 3, 10),
    verbose: bool = True,
) -> Dict[str, float]:

    model.eval()
    dataset = loader.valid_set if split == "valid" else loader.test_set
    dl = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, num_workers=0)

    mrr_sum = 0.0
    hits    = {k: 0.0 for k in hits_at}
    total   = 0

    for batch in tqdm(dl, desc=f"Eval [{split}]", disable=not verbose):
        subs, rels, objs, times = [x.to(device) for x in batch]

        h_rels, h_objs, h_times, h_mask = loader.build_history_batch(
            subs.cpu().tolist(), rels.cpu().tolist(),
            times.cpu().tolist(), max_hist,
        )
        h_rels  = h_rels.to(device)
        h_objs  = h_objs.to(device)
        h_times = h_times.to(device)
        h_mask  = h_mask.to(device)

        copy_sc = loader.get_copy_scores_batch(
            subs.cpu().tolist(), rels.cpu().tolist(), times.cpu().tolist()
        ).to(device)

        logits = model(subs, rels, h_rels, h_objs, h_times,
                       times, h_mask, copy_sc)                 # (B, N)

        for i in range(subs.size(0)):
            s, r, t = subs[i].item(), rels[i].item(), times[i].item()
            o_true  = objs[i].item()
            sc      = logits[i].clone()

            # filter known true answers
            for o_filter in loader.index.all_answers.get((s, r, t), set()):
                if o_filter != o_true:
                    sc[o_filter] = float("-inf")

            rank = (sc > sc[o_true]).sum().item() + 1
            mrr_sum += 1.0 / rank
            for k in hits_at:
                hits[k] += float(rank <= k)
            total += 1

    results = {"MRR": mrr_sum / total}
    results.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    return results
