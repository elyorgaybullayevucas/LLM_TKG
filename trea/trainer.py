"""
AURORA-TKG Trainer — per-epoch metrics table, checkpointing.
"""
import os, time, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from trea.config import AURORAConfig
from trea.model import AURORAModel
from trea.loss import AURORALoss
from trea.data import TKGDataLoader, aurora_collate
from trea.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


HEADER = (
    f"{'Ep':>4} │ {'Time':>6} │ "
    f"{'Loss':>8} {'CE':>8} {'NCE':>8} │ "
    f"{'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} │ "
    f"{'LR':>9}"
)
SEP = "─" * len(HEADER)


def _row(ep, t, ld, m, lr, best=False):
    star = "★" if best else " "
    return (
        f"{star}{ep:>3} │ {t:>5.1f}s │ "
        f"{ld['total']:>8.4f} {ld['ce']:>8.4f} {ld['infonce']:>8.4f} │ "
        f"{m.get('MRR',0):>7.4f} {m.get('Hits@1',0):>7.4f} "
        f"{m.get('Hits@3',0):>7.4f} {m.get('Hits@10',0):>7.4f} │ "
        f"{lr:>9.2e}"
    )


class AURORATrainer:

    def __init__(self, cfg: AURORAConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # ── GPU ni DARHOL egallash ────────────────────────────────────────────
        use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
            self.device = torch.device("cuda:0")
            gpu_info    = torch.cuda.get_device_name(0)
            total_mem   = torch.cuda.get_device_properties(0).total_memory
            free_mem    = torch.cuda.mem_get_info()[0]
            # 85% ni band qilamiz — boshqalar egallay olmasin
            reserve_gb  = int(free_mem * 0.85) // (1024**3)
            reserve_n   = reserve_gb * (1024**3) // 4  # float32 elements
            print(f"[Device] GPU {cfg.gpu} — {gpu_info}")
            print(f"[GPU]    {free_mem/1024**3:.1f} GB bo'sh → "
                  f"{reserve_gb} GB band qilinmoqda …")
            _placeholder = torch.zeros(reserve_n, device=self.device)
            print(f"[GPU]    Joy band qilindi ✓")
        else:
            self.device = torch.device("cpu")
            _placeholder = None
            print("[Device] CPU")

        # ── data (CPU da pre-computation) ────────────────────────────────────
        self.loader = TKGDataLoader(cfg)

        # ── model GPU ga ──────────────────────────────────────────────────────
        # Placeholder ni bo'shatamiz — model + training shu xotirani ishlatadi
        if _placeholder is not None:
            del _placeholder
            torch.cuda.empty_cache()

        self.model = AURORAModel(
            num_entities=self.loader.num_entities,
            num_relations=self.loader.num_relations,
            cfg=cfg,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        print(f"[Model] AURORA-TKG  params={n_params:,}")

        # AMP scaler (bfloat16 on A100 — native hardware support)
        self.use_amp = (self.device.type == "cuda")
        self.scaler  = GradScaler(enabled=self.use_amp)

        # loss / optim / scheduler
        self.loss_fn = AURORALoss(
            smoothing=cfg.label_smoothing,
            alpha=cfg.alpha_infonce,
            temperature=cfg.infonce_temp,
        )
        self.optim = AdamW(self.model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = CosineAnnealingLR(self.optim, T_max=cfg.epochs,
                                       eta_min=cfg.lr * 0.01)

        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,  exist_ok=True)
        self.log_path = os.path.join(cfg.log_dir, f"{cfg.dataset}_log.jsonl")
        self.best_mrr = 0.0
        self.best_ep  = 0

    # ── one training epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        cfg = self.cfg

        dl = DataLoader(
            self.loader.train_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True, collate_fn=aurora_collate,
            persistent_workers=True,
        )

        tot_total = tot_ce = tot_nce = 0.0
        n = 0

        for batch in tqdm(dl, desc=f" ep{epoch}", leave=False,
                          dynamic_ncols=True):
            (subs, rels, objs, times,
             ne, nr, nm, rel_copy, ent_copy) = batch

            subs     = subs.to(self.device)
            rels     = rels.to(self.device)
            objs     = objs.to(self.device)
            times    = times.to(self.device)
            ne       = ne.to(self.device)
            nr       = nr.to(self.device)
            nm       = nm.to(self.device)
            # pad copy scores to num_entities
            N = self.loader.num_entities
            if rel_copy.shape[1] < N:
                pad = torch.zeros(rel_copy.shape[0], N - rel_copy.shape[1])
                rel_copy = torch.cat([rel_copy, pad], dim=1)
            if ent_copy.shape[1] < N:
                pad = torch.zeros(ent_copy.shape[0], N - ent_copy.shape[1])
                ent_copy = torch.cat([ent_copy, pad], dim=1)
            rel_copy = rel_copy.to(self.device)
            ent_copy = ent_copy.to(self.device)

            with autocast(dtype=torch.bfloat16, enabled=self.use_amp):
                logits     = self.model(subs, rels, ne, nr, nm,
                                        rel_copy, ent_copy)
                query_repr = self.model.encode_query(subs, rels, ne, nr, nm)
                loss_d     = self.loss_fn(logits, objs, query_repr,
                                          self.model.ent_emb)

            self.optim.zero_grad()
            self.scaler.scale(loss_d["_total"]).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            cfg.grad_clip)
            self.scaler.step(self.optim)
            self.scaler.update()

            tot_total += loss_d["total"]
            tot_ce    += loss_d["ce"]
            tot_nce   += loss_d["infonce"]
            n         += 1

        self.sched.step()
        return {"total": tot_total / n, "ce": tot_ce / n, "infonce": tot_nce / n}

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"\n{'═'*len(HEADER)}")
        print(f"  AURORA-TKG │ {cfg.dataset} │ epochs={cfg.epochs} │ "
              f"d={cfg.embed_dim} │ H={cfg.long_len} │ S={cfg.short_len} │ "
              f"K={cfg.k_neighbors} │ α={cfg.alpha_infonce}")
        print(f"{'═'*len(HEADER)}\n")
        print(HEADER)
        print(SEP)

        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()

            loss_d  = self._train_epoch(ep)
            elapsed = time.time() - t0
            lr_now  = self.sched.get_last_lr()[0]

            metrics = {}
            if ep % cfg.eval_every == 0:
                metrics = evaluate(
                    self.model, self.loader, split="valid",
                    device=self.device, batch_size=cfg.batch_size,
                    hits_at=list(cfg.hits_at), verbose=False,
                )

            is_best = metrics.get("MRR", 0) > self.best_mrr
            if is_best:
                self.best_mrr = metrics["MRR"]
                self.best_ep  = ep
                self._save("best")
            if ep % 5 == 0:
                self._save(f"ep{ep:03d}")

            print(_row(ep, elapsed, loss_d, metrics, lr_now, is_best))

            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": ep, "loss": loss_d,
                    "metrics": metrics, "lr": lr_now,
                }) + "\n")

        # ── final test ─────────────────────────────────────────────────────────
        print(SEP)
        print(f"\n★  Best valid MRR = {self.best_mrr:.4f}  (epoch {self.best_ep})")
        print("\nLoading best checkpoint → TEST evaluation …")
        self._load("best")

        test_m = evaluate(
            self.model, self.loader, split="test",
            device=self.device, batch_size=cfg.batch_size,
            hits_at=list(cfg.hits_at), verbose=True,
        )

        print(f"\n{'═'*50}")
        print(f"  TEST RESULTS — {cfg.dataset}")
        print(f"{'═'*50}")
        for k, v in test_m.items():
            print(f"  {k:<12}: {v:.4f}")
        print(f"{'═'*50}")

        out = os.path.join(cfg.save_dir, f"{cfg.dataset}_results.json")
        with open(out, "w") as f:
            json.dump({"dataset": cfg.dataset, "best_epoch": self.best_ep,
                       "valid_mrr": self.best_mrr, "test": test_m,
                       "config": cfg.__dict__}, f, indent=2, default=str)
        print(f"Saved → {out}")
        return test_m

    def _save(self, tag: str):
        p = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_{tag}.pt")
        torch.save({"model": self.model.state_dict(),
                    "optim": self.optim.state_dict()}, p)

    def _load(self, tag: str):
        p = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_{tag}.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=self.device)
            self.model.load_state_dict(ck["model"])
            print(f"  Loaded: {p}")
