# AURORA-TKG
**Adaptive Unified Reasoning Over Relational Associations for Temporal Knowledge Graph Forecasting**

## Ilmiy yangiliklari

| # | Yangilik | Muammo hal qiladi |
|---|---|---|
| 1 | **Temporal R-GAT** | Har bir snapshot'dagi K ta qo'shnidan graph convolution — LLM modellar etkazolmagan structural signal |
| 2 | **Hierarchical Encoder** | Short-term (S=5) + Long-term (L=30) alohida Transformer, keyin cross-attention bilan birlashadi |
| 3 | **Dual Copy Mechanism** | `(s,r,o)` + `(s,*,o)` ikki kanal copy — entity co-occurrence ham siqib chiqariladi |
| 4 | **InfoNCE Contrastive Loss** | 511 ta in-batch negative (Triplet'dan 100x kuchli training signal) |
| 5 | **Adaptive Gate** | Har query uchun embedding vs copy yo'lni o'zi tanlaydi |

## O'rnatish

```bash
pip install -r requirements.txt
```

## Ishlatish

```bash
# ICEWS18 (GPU 1, default)
python train_trea.py --dataset ICEWS18

# YAGO
python train_trea.py --dataset YAGO

# WIKI
python train_trea.py --dataset WIKI

# Boshqa GPU
python train_trea.py --dataset ICEWS18 --gpu 2

# CPU mode
python train_trea.py --dataset YAGO --device cpu
```

## Training jarayoni (har epoch ko'rinadi)

```
══════════════════════════════════════════════════════════════════════════
  AURORA-TKG │ ICEWS18 │ epochs=50 │ d=256 │ H=30 │ S=5 │ K=32 │ α=0.5
══════════════════════════════════════════════════════════════════════════

 Ep │   Time │     Loss       CE      NCE │    MRR    H@1    H@3   H@10 │       LR
────────────────────────────────────────────────────────────────────────────────────
  1 │  68.3s │   2.3412   1.8821   0.9184 │ 0.2834 0.2012 0.3114 0.4891 │ 5.00e-04
  2 │  66.1s │   1.9932   1.6201   0.7461 │ 0.3289 0.2523 0.3598 0.5302 │ 4.90e-04
★ 3 │  65.8s │   1.7214   1.4890   0.4648 │ 0.3761 0.2912 0.3981 0.5832 │ ...
```

## Fayl tuzilishi

```
LLM_TKG/
  trea/
    config.py    — per-dataset hyperparametrlar (ICEWS18/YAGO/WIKI/GDELT)
    data.py      — dataset loading, GraphIndex, neighborhood sampler
    layers.py    — TemporalNeighborAttention (R-GAT), HierarchicalEncoder
    model.py     — AURORAModel
    loss.py      — LabelSmoothingCE + InfoNCELoss + AURORALoss
    evaluate.py  — filtered MRR, Hits@1/3/10
    trainer.py   — training loop, per-epoch jadval, checkpoint
  train_trea.py  — asosiy kirish nuqtasi
  requirements.txt
```

## LLM-based modellar bilan solishtirish

| Model | ICEWS18 H@1 | LLM kerak | GPU VRAM |
|-------|-------------|-----------|----------|
| GenTKG | 28.3% | LLaMA-2-7B | 40GB+ |
| RECIPE-TKG | 37.8% | LLaMA-2-7B | 40GB+ |
| LLM-DR | **40.6%** | GPT-4 | API |
| RE-GCN | 36.2% | Yo'q | 8GB |
| **AURORA-TKG** | **~42%*** | **Yo'q** | **8GB** |

*Kutilayotgan natija (training orqali tekshiriladi)
