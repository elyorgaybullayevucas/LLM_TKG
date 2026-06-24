"""
Filtered evaluation: MRR, Hits@1, Hits@3, Hits@10.

Standard TKG forecasting protocol:
  For each (s, r, o_true, t):
    1. Score all entities.
    2. Set scores of OTHER known-true answers to -inf (filter).
    3. Rank o_true; accumulate MRR and Hits@k.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm

from trea.model import AURORAModel
from trea.data import TKGDataLoader


@torch.no_grad()
def evaluate(
    model:      AURORAModel,
    loader:     TKGDataLoader,
    split:      str   = "valid",
    device:     torch.device = torch.device("cpu"),
    batch_size: int   = 256,
    hits_at:    tuple = (1, 3, 10),
    verbose:    bool  = True,
) -> Dict[str, float]:

    model.eval()
    dataset = loader.valid_set if split == "valid" else loader.test_set
    cfg     = loader.cfg
    dl      = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    mrr_sum = 0.0
    hits    = {k: 0.0 for k in hits_at}
    total   = 0

    for batch in tqdm(dl, desc=f"Eval [{split}]", disable=not verbose,
                      dynamic_ncols=True):
        subs, rels, objs, times = [x.to(device) for x in batch]

        sl = subs.cpu().tolist()
        rl = rels.cpu().tolist()
        tl = times.cpu().tolist()

        ne, nr, nm = loader.build_neighborhood_batch(sl, tl)
        ne, nr, nm = ne.to(device), nr.to(device), nm.to(device)

        rel_copy = loader.get_rel_copy_batch(sl, rl, tl).to(device)
        ent_copy = loader.get_ent_copy_batch(sl, tl).to(device)

        logits = model(subs, rels, ne, nr, nm, rel_copy, ent_copy)   # (B, N)

        for i in range(subs.size(0)):
            s, r, t = subs[i].item(), rels[i].item(), times[i].item()
            o_true  = objs[i].item()
            sc      = logits[i].clone()

            for o_f in loader.index.all_answers.get((s, r, t), set()):
                if o_f != o_true:
                    sc[o_f] = float("-inf")

            rank = (sc > sc[o_true]).sum().item() + 1
            mrr_sum += 1.0 / rank
            for k in hits_at:
                hits[k] += float(rank <= k)
            total += 1

    results = {"MRR": mrr_sum / total}
    results.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    return results
