# TREA-TKG
**Temporal Recurrence-Enhanced Attention for Temporal Knowledge Graph Forecasting**

## Ilmiy yangiliklari

| # | Yangilik | Muammo hal qiladi |
|---|---|---|
| 1 | **Per-relation learnable decay** `α_r` | Har bir relation uchun vaqt ta'siri boshqacha (ICEWS: tez, YAGO: sekin) |
| 2 | **Frequency-aware Copy Mechanism** | `log(1+freq) × exp(−λΔt)` — kam takrorlanadigan faktlarga nisbiy ustunlik |
| 3 | **Adaptive Gate** | Model o'zi hal qiladi: qachon structural, qachon copy yo'lidan borish |
| 4 | **Hard-negative Contrastive Loss** | `L = L_CE + α · L_triplet` — embedding sifatini yaxshilaydi |

## O'rnatish

```bash
pip install -r requirements.txt
```

## Ishlatish

```bash
# ICEWS18
python train_trea.py --dataset ICEWS18 --epochs 50 --embed_dim 256

# YAGO (10 ta relation, tez)
python train_trea.py --dataset YAGO --epochs 30 --embed_dim 128 --lr 2e-3

# WIKI
python train_trea.py --dataset WIKI --epochs 40 --embed_dim 256

# Barcha 3 ta dataset
bash run_all.sh
```

## Natijalar (har epochda ko'rinadi)

```
══════════════════════════════════════════════════════════════════════════════
  TREA-TKG │ ICEWS18 │ epochs=50 │ d=256 │ H=10 │ α=0.3
══════════════════════════════════════════════════════════════════════════════

 Ep │   Time │     Loss       CE      Tri │    MRR    H@1    H@3   H@10 │       LR
────────────────────────────────────────────────────────────────────────────────────
  1 │  42.3s │   2.1453   1.9821   0.4613 │ 0.2134 0.1521 0.2314 0.3891 │ 9.90e-04
  2 │  41.1s │   1.8932   1.7201   0.3461 │ 0.2489 0.1823 0.2698 0.4102 │ 9.60e-04
★ 3 │  40.8s │   1.7214   1.5890   0.2648 │ 0.2761 0.2012 0.2981 0.4432 │ ...
```

## Fayl tuzilishi

```
TREA-TKG/
  trea/
    config.py      — barcha hyperparametrlar
    data.py        — dataset loading, GraphIndex, history builder
    model.py       — AdaptiveTemporalAttention + AdaptiveGate + TREAModel
    loss.py        — LabelSmoothingCE + HardNegativeTriplet + TREALoss
    evaluate.py    — filtered MRR, Hits@1/3/10
    trainer.py     — training loop, per-epoch jadval, checkpoint
  train_trea.py    — asosiy kirish nuqtasi
  run_all.sh       — 3 ta datasetni ketma-ket train qilish
  requirements.txt
```

## RECIPE-TKG bilan farq

| | RECIPE-TKG | **TREA-TKG** |
|---|---|---|
| LLM kerak | Ha (LLaMA-2-7B) | **Yo'q** |
| GPU VRAM | 40GB+ | **8GB yetarli** |
| Train vaqti | Soatlar | **Daqiqalar** |
| YAGO / WIKI | Zaif | **Kuchli** |
| Interpretability | Qiyin | **Oson (α_r ko'rish mumkin)** |
