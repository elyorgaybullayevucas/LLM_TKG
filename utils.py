import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import bitsandbytes as bnb

from datasets import load_dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------

def get_model_and_tokenizer(args, config):
    if args.BIT_8:
        qcfg8 = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, quantization_config=qcfg8,
            device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    elif args.BIT_4:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, quantization_config=qcfg,
            device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16)

    if "Llama-3" in args.MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.MODEL_NAME, trust_remote_code=True, pad_token="</s>")

    model = get_peft_model(model, config)

    embeddings = model.get_input_embeddings()
    if hasattr(embeddings, "weight"):
        embeddings.weight.requires_grad = True

    model.config.use_cache = False
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def get_lora_config(args):
    target_modules = (
        ["q_proj", "k_proj", "v_proj", "o_proj"]
        if "Llama-3" in args.MODEL_NAME else
        ["q_proj", "v_proj"]
    )
    return LoraConfig(
        r=args.LORA_R, lora_alpha=args.LORA_ALPHA,
        lora_dropout=args.LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _llama2_tokenize(args, tokenizer, data_type, dp):
    if data_type == "json":
        src = tokenizer(dp["context"], max_length=args.CONTEXT_LEN,
                        padding="max_length", truncation=True)
        tgt = tokenizer(dp["target"], max_length=args.TARGET_LEN,
                        padding=False, truncation=True)
        ids = src["input_ids"] + tgt["input_ids"] + [tokenizer.eos_token_id] + \
              [2] * (args.TARGET_LEN - len(tgt["input_ids"]))
        mask = src["attention_mask"] + tgt["attention_mask"] + [1] + \
               [0] * (args.TARGET_LEN - len(tgt["input_ids"]))
        labels = [-100] * args.CONTEXT_LEN + tgt["input_ids"] + \
                 [tokenizer.eos_token_id] + [-100] * (args.TARGET_LEN - len(tgt["input_ids"]))
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    else:
        out = tokenizer(dp["text"], max_length=args.TEXT_LEN,
                        padding="max_length", truncation=True)
        out["input_ids"].extend([tokenizer.eos_token_id])
        out["attention_mask"].extend([1])
        return out


def _llama3_tokenize(args, tokenizer, data_type, dp):
    if data_type == "json":
        full = dp["context"] + dp["target"]
        tok = tokenizer(full, max_length=args.CONTEXT_LEN + args.TARGET_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        ans_start = len(tokenizer.encode(dp["context"]))
        labels = tok["input_ids"].clone()
        labels[:, :ans_start] = -100
        return {"input_ids": tok["input_ids"][0],
                "attention_mask": tok["attention_mask"][0],
                "labels": labels[0]}
    else:
        tok = tokenizer(dp["text"], max_length=args.TEXT_LEN,
                        padding="max_length", truncation=True)
        tok["input_ids"].append(tokenizer.eos_token_id)
        tok["attention_mask"].append(1)
        return tok


def process_data(args, tokenizer, data_type, dataset):
    fn = _llama3_tokenize if "Llama-3" in args.MODEL_NAME else _llama2_tokenize
    return dataset.shuffle().map(
        lambda dp: fn(args, tokenizer, data_type, dp), batched=False)


def get_dataset(data_type, data_path):
    if data_type == "json":
        return load_dataset("json", data_files=data_path)
    return load_dataset("text", data_files=data_path)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SelfAttentionPooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, x):
        x = x.to(dtype=self.q.weight.dtype)
        scores = torch.matmul(self.q(x), self.k(x).transpose(-2, -1)) / (x.size(-1) ** 0.5)
        return torch.matmul(F.softmax(scores, dim=1), self.v(x)).mean(dim=1)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with learnable temperature.

    Replaces the original fixed-margin triplet loss. Benefits:
    - Uses ALL negatives per anchor (not just hardest)
    - Temperature adapts per dataset during training
    - More stable gradients for sparse graphs (YAGO/WIKI)
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.log_temp = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float32))

    @property
    def temperature(self):
        return torch.clamp(torch.exp(self.log_temp), min=0.01, max=1.0)

    def _infonce(self, anchor, positives, negatives):
        anchor = F.normalize(anchor.unsqueeze(0), dim=-1)        # (1, D)
        positives = F.normalize(positives, dim=-1)                # (P, D)
        negatives = F.normalize(negatives, dim=-1)                # (N, D)
        t = self.temperature
        pos_logits = (anchor @ positives.T).squeeze(0) / t       # (P,)
        neg_logits = (anchor @ negatives.T).squeeze(0) / t       # (N,)
        # Each positive competes against all negatives
        logits = torch.cat(
            [pos_logits.unsqueeze(1),
             neg_logits.unsqueeze(0).expand(len(pos_logits), -1)], dim=1)  # (P, N+1)
        labels = torch.zeros(len(pos_logits), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    def forward(self, anchors, positives, negatives, lm_outputs):
        cl_loss = torch.tensor(0.0, device=anchors[0].device)
        valid = 0
        for a, p, n in zip(anchors, positives, negatives):
            if p is None or n is None or p.numel() == 0 or n.numel() == 0:
                continue
            try:
                a.requires_grad_(True)
                cl_loss = cl_loss + self._infonce(a, p, n)
                valid += 1
            except Exception:
                continue
        ce_loss = lm_outputs.loss
        if valid == 0:
            return ce_loss, cl_loss, ce_loss
        cl_loss = cl_loss / valid
        return self.alpha * cl_loss + (1 - self.alpha) * ce_loss, cl_loss, ce_loss


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_sentiment(data_path: str) -> Dict[str, str]:
    parent = os.path.dirname(data_path)
    path = os.path.join(parent, "relation2sentiment.txt")
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def _truncate_decode(tokens, eos_id, tokenizer):
    out = []
    for row in tokens:
        pos = (row == eos_id).nonzero(as_tuple=True)[0]
        end = pos[0].item() if len(pos) else len(row)
        out.append(tokenizer.decode(row[:end].tolist(), skip_special_tokens=True))
    return out


def _extract_brackets(strings):
    import re
    results = []
    for s in strings:
        m = re.search(r'\n\n<</SYS>>(.*)', s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        brackets = re.findall(r'\[(.*?)\]', s)
        triples = []
        for c in brackets:
            fc, lc = c.find(','), c.rfind(',')
            if fc != -1 and lc != -1 and fc != lc:
                triples.append([c[:fc].strip(),
                                 c[fc+1:lc].strip(),
                                 c[lc+1:].strip()])
        results.append(triples)
    return results


def _extract_target(texts, tokenizer):
    import re
    decoded = tokenizer.batch_decode(texts.tolist(), skip_special_tokens=False)
    out = []
    for t in decoded:
        t = t.split('<s>')[-1].strip().split('</s>')[0].strip()
        parts = t.split('.', 1)
        out.append(parts[1] if len(parts) > 1 else t)
    return out


def _batch_embed(aggregator, tokenizer, model, texts):
    tokens = tokenizer.batch_encode_plus(texts, return_tensors='pt',
                                         padding=True, truncation=True)
    embs = model.get_input_embeddings()(tokens['input_ids'].to(model.device))
    return aggregator(embs)


def get_sample_embeddings(attn_pool, tokenizer, model, data_path, inputs):
    sentiment = _load_sentiment(data_path)
    decoded = _truncate_decode(inputs["input_ids"], tokenizer.eos_token_id, tokenizer)
    triples_batch = _extract_brackets(decoded)
    targets = _extract_target(inputs["input_ids"], tokenizer)

    flat = []
    for i, triples in enumerate(triples_batch):
        for s, r, o in triples[:-1]:
            o_clean = o.split('.', 1)[1] if '.' in o else o
            flat.append((s, r, o_clean))
        if triples:
            qs, qr, _ = triples[-1]
            flat.append((qs, qr, targets[i]))

    groups = defaultdict(lambda: {"positive": [], "negative": []})
    for s, r, o in flat:
        sent = sentiment.get(r)
        if sent == "positive":
            groups[s]["positive"].append(o)
        elif sent == "negative":
            groups[s]["negative"].append(o)

    for s in groups:
        pos_set = set(groups[s]["positive"])
        neg_set = set(groups[s]["negative"])
        conflict = pos_set & neg_set
        groups[s]["positive"] = [o for o in groups[s]["positive"] if o not in conflict]
        groups[s]["negative"] = [o for o in groups[s]["negative"] if o not in conflict]

    if not groups:
        return None, None, None

    all_entities = list({e for s, r, o in flat for e in (s, o)})
    vecs = _batch_embed(attn_pool, tokenizer, model, [str(e) for e in all_entities])
    emb_map = {e: vecs[i] for i, e in enumerate(all_entities)}

    anchors, positives, negatives = [], [], []
    for subj, obj_dict in groups.items():
        if subj not in emb_map:
            continue
        p = [emb_map[o] for o in obj_dict["positive"] if o in emb_map]
        n = [emb_map[o] for o in obj_dict["negative"] if o in emb_map]
        anchors.append(emb_map[subj])
        positives.append(torch.stack(p) if p else None)
        negatives.append(torch.stack(n) if n else None)

    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

class _BaseTrainer(transformers.Trainer):
    def __init__(self, model, ckpt="", data_path="", **kwargs):
        super().__init__(model=model, **kwargs)
        self.ckpt = ckpt
        self.data_path = data_path


class PlainTrainer(_BaseTrainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        loss = model(input_ids=inputs["input_ids"], labels=inputs["labels"]).loss
        print(loss)
        return loss


class ContrastiveTrainer(_BaseTrainer):
    def __init__(self, model, tokenizer=None, ckpt="", data_path="",
                 weight=0.4, temperature=0.07, **kwargs):
        super().__init__(model=model, ckpt=ckpt, data_path=data_path, **kwargs)
        self.tokenizer = tokenizer
        self.attn_pool = SelfAttentionPooling(model.config.hidden_size).to(self.model.device)
        self.loss_fn = InfoNCELoss(temperature=temperature, alpha=weight)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"],
                        output_hidden_states=True)
        self.attn_pool.to(outputs.hidden_states[-1].device)
        anchors, positives, negatives = get_sample_embeddings(
            self.attn_pool, self.tokenizer, model, self.data_path, inputs)
        if anchors is None:
            return outputs.loss
        total, cl, ce = self.loss_fn(anchors, positives, negatives, outputs)
        print(f"cl={cl.item():.4f}  ce={ce.item():.4f}  total={total.item():.4f}  "
              f"temp={self.loss_fn.temperature.item():.4f}")
        return total

    def create_optimizer(self):
        self.optimizer = bnb.optim.AdamW8bit(
            list(self.model.parameters()) +
            list(self.attn_pool.parameters()) +
            list(self.loss_fn.parameters()),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=0.01)
        if self.ckpt:
            opt_ckpt = os.path.join(self.ckpt, "optimizer.pt")
            if os.path.exists(opt_ckpt):
                try:
                    self.optimizer.load_state_dict(
                        torch.load(opt_ckpt, map_location=self.model.device))
                except Exception as e:
                    print(f"[WARN] optimizer ckpt load failed: {e}")
        return self.optimizer

    def _wrap_model(self, *args, **kwargs):
        model = super()._wrap_model(*args, **kwargs)
        self.create_optimizer()
        return model


def get_trainer(args, model, data, tokenizer):
    ga = args.BATCH_SIZE // args.MICRO_BATCH_SIZE
    load_best = args.LOAD_BEST_MODEL_AT_END == 1
    tr_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.MICRO_BATCH_SIZE,
        gradient_accumulation_steps=ga,
        warmup_ratio=0.05,
        num_train_epochs=args.EPOCHS,
        learning_rate=args.LEARNING_RATE,
        save_strategy="steps",
        save_steps=args.SAVE_STEPS,
        eval_steps=args.EVAL_STEPS,
        output_dir=args.OUTPUT_DIR,
        save_total_limit=args.SAVE_TOTAL_LIMIT,
        eval_strategy=args.EVAL_STRATEGY,
        report_to=args.REPORT_TO,
        run_name=args.RUN_NAME,
        load_best_model_at_end=load_best,
        logging_strategy="steps",
        logging_steps=args.LOGGING_STEPS,
        fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.95,
        gradient_checkpointing=True,
    )
    collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    common = dict(train_dataset=data["train"], args=tr_args,
                  data_collator=collator, tokenizer=tokenizer,
                  ckpt=args.RESUME_CKPT, data_path=args.DATA_PATH)
    if args.CONTRASTIVE == 0:
        return PlainTrainer(model=model, **common)
    return ContrastiveTrainer(
        model=model,
        weight=args.CONTRASTIVE_WEIGHT,
        temperature=getattr(args, "CONTRASTIVE_TEMP", 0.07),
        **common)
