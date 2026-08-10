import json, random, os, datetime, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF

# install: pip install -q transformers accelerate stanza pytorch-crf scikit-learn
# runs 3 variants (full, no_crf, no_bilstm) across 5 folds, plus a fold 1 detail dump

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "encoder_name": "xlm-roberta-base",
    "lstm_hidden": 256,
    "pos_dim": 32,
    "dropout": 0.3,
    "batch_size": 16,
    "epochs": 20,
    "patience": 4,
    "lr_encoder": 2e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "max_words": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
N_FOLDS = 5
DEV_FRAC_WITHIN_TRAIN = 0.15

ABLATION_VARIANTS = {
    "full":       {"use_bilstm": True,  "use_crf": True},
    "no_crf":     {"use_bilstm": True,  "use_crf": False},
    "no_bilstm":  {"use_bilstm": False, "use_crf": True},
}

# which variant and fold gets a full per-sentence prediction dump
DETAIL_DUMP_TARGET = ("full", 1)

print("Device:", CONFIG["device"])

from google.colab import drive
drive.mount('/content/drive')

RESULTS_DIR = "/content/drive/MyDrive/english_ner_ablation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Results will be saved to:", RESULTS_DIR)

from google.colab import files
print("Upload your English gold standard JSON file:")
uploaded = files.upload()
DATA_PATH = list(uploaded.keys())[0]
print("Loaded file:", DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Total documents: {len(raw_data)}")
all_labels = sorted({e["label"] for d in raw_data for e in d["entities"]})
print("Entity types found:", all_labels)

raw_by_id = {item["id"]: item for item in raw_data}

import stanza
stanza.download('en', verbose=False)
nlp = stanza.Pipeline('en', processors='tokenize,pos', verbose=False,
                       use_gpu=(CONFIG["device"] == "cuda"))

def get_tokens_with_offsets(text):
    doc = nlp(text)
    tokens = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            upos = tok.words[0].upos if tok.words else "X"
            tokens.append({"text": tok.text, "start": tok.start_char, "end": tok.end_char, "pos": upos})
    return tokens

def entities_to_bioes(tokens, entities):
    tags = ["O"] * len(tokens)
    for ent in entities:
        e_start, e_end, label = ent["start"], ent["end"], ent["label"]
        idxs = [i for i, t in enumerate(tokens) if t["start"] < e_end and t["end"] > e_start]
        if not idxs:
            continue
        if len(idxs) == 1:
            tags[idxs[0]] = f"S-{label}"
        else:
            tags[idxs[0]] = f"B-{label}"
            tags[idxs[-1]] = f"E-{label}"
            for i in idxs[1:-1]:
                tags[i] = f"I-{label}"
    return tags

def process_dataset(raw_data):
    processed = []
    skipped = 0
    for item in raw_data:
        text = item["text"]
        if not text.strip():
            skipped += 1
            processed.append(None)
            continue
        tokens = get_tokens_with_offsets(text)
        if not tokens:
            skipped += 1
            processed.append(None)
            continue
        tags = entities_to_bioes(tokens, item["entities"])
        processed.append({
            "id": item["id"], "words": [t["text"] for t in tokens], "pos": [t["pos"] for t in tokens],
            "tags": tags, "starts": [t["start"] for t in tokens], "ends": [t["end"] for t in tokens],
        })
    print(f"Processed {len(processed) - skipped} docs, skipped {skipped}")
    return processed, skipped

processed_data, n_skipped = process_dataset(raw_data)
# a skip would break KFold positional alignment, so this must be zero
assert n_skipped == 0, f"{n_skipped} document(s) skipped, fix before proceeding."
print(processed_data[0])

all_tags = sorted({t for d in processed_data for t in d["tags"]})
tag2id = {t: i for i, t in enumerate(all_tags)}
id2tag = {i: t for t, i in tag2id.items()}

all_pos = sorted({p for d in processed_data for p in d["pos"]})
pos2id = {p: i + 1 for i, p in enumerate(all_pos)}
pos2id["<PAD>"] = 0

print(f"Num tags: {len(tag2id)} | Num POS tags: {len(pos2id)}")

# same fold construction as the original English LoRA script
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_indices = list(range(len(processed_data)))
folds = list(kf.split(all_indices))
print(f"Built {N_FOLDS} folds, about {len(folds[0][0])} train and {len(folds[0][1])} held out per fold")

seen = set()
for fi, (_, test_idx) in enumerate(folds):
    assert not (seen & set(test_idx)), f"Fold {fi} overlaps a previous fold."
    seen.update(test_idx)
assert len(seen) == len(processed_data)
print("Fold coverage check passed.")

# per-fold dataset stats, no model needed, useful for explaining any weak fold
print("\nPer-fold test set statistics")
fold_stats = {}
for fold_idx, (_, test_idx) in enumerate(folds):
    docs = [processed_data[i] for i in test_idx]
    n_docs = len(docs)
    tok_counts = [len(d["words"]) for d in docs]
    ent_counts = [sum(1 for t in d["tags"] if t.startswith(("B-", "S-"))) for d in docs]
    avg_tokens = np.mean(tok_counts)
    avg_entities = np.mean(ent_counts)
    entity_density = sum(ent_counts) / sum(tok_counts)
    zero_entity_docs = sum(1 for c in ent_counts if c == 0)

    fold_stats[fold_idx] = {
        "n_docs": n_docs,
        "avg_tokens_per_doc": round(float(avg_tokens), 2),
        "max_tokens": int(max(tok_counts)),
        "avg_entities_per_doc": round(float(avg_entities), 2),
        "entity_density_per_token": round(float(entity_density), 4),
        "zero_entity_docs": zero_entity_docs,
    }
    print(f"Fold {fold_idx}: n={n_docs} | avg_tokens={avg_tokens:.1f} (max {max(tok_counts)}) "
          f"| avg_entities={avg_entities:.2f} | entity_density={entity_density:.4f} "
          f"| zero-entity docs={zero_entity_docs}")

with open(os.path.join(RESULTS_DIR, "fold_dataset_statistics.json"), "w", encoding="utf-8") as f:
    json.dump(fold_stats, f, ensure_ascii=False, indent=2)
print(f"Saved to: {os.path.join(RESULTS_DIR, 'fold_dataset_statistics.json')}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["encoder_name"])

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, pos2id, max_words):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.pos2id = pos2id
        self.max_words = max_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["words"][:self.max_words]
        pos = item["pos"][:self.max_words]
        tags = item["tags"][:self.max_words]

        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                              max_length=256, return_tensors=None)
        word_ids = enc.word_ids()

        first_subword_idx = []
        seen_w = set()
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in seen_w:
                first_subword_idx.append(i)
                seen_w.add(wid)

        return {
            "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "first_subword_idx": first_subword_idx,
            "pos_ids": [self.pos2id.get(p, 0) for p in pos],
            "tag_ids": [self.tag2id[t] for t in tags],
            "num_words": len(words),
        }

def collate_fn(batch):
    max_subword_len = max(len(b["input_ids"]) for b in batch)
    max_word_len = max(b["num_words"] for b in batch)

    input_ids = torch.zeros(len(batch), max_subword_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_subword_len, dtype=torch.long)
    first_subword_idx = torch.zeros(len(batch), max_word_len, dtype=torch.long)
    word_mask = torch.zeros(len(batch), max_word_len, dtype=torch.bool)
    pos_ids = torch.zeros(len(batch), max_word_len, dtype=torch.long)
    tag_ids = torch.zeros(len(batch), max_word_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        attn_mask[i, :L] = torch.tensor(b["attention_mask"])
        W = b["num_words"]
        idxs = b["first_subword_idx"][:W]
        first_subword_idx[i, :len(idxs)] = torch.tensor(idxs)
        word_mask[i, :W] = True
        pos_ids[i, :W] = torch.tensor(b["pos_ids"][:W])
        tag_ids[i, :W] = torch.tensor(b["tag_ids"][:W])

    return {
        "input_ids": input_ids, "attention_mask": attn_mask,
        "first_subword_idx": first_subword_idx, "word_mask": word_mask,
        "pos_ids": pos_ids, "tag_ids": tag_ids,
    }

# unified model, supports full, no_crf, no_bilstm via flags
class NERModel(nn.Module):
    def __init__(self, encoder_name, num_tags, num_pos, lstm_hidden, pos_dim, dropout,
                 use_bilstm=True, use_crf=True):
        super().__init__()
        self.use_bilstm = use_bilstm
        self.use_crf = use_crf
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        self.pos_embed = nn.Embedding(num_pos, pos_dim, padding_idx=0)
        feat_dim = enc_dim + pos_dim

        if use_bilstm:
            self.bilstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True, bidirectional=True, num_layers=1)
            classifier_in = lstm_hidden * 2
        else:
            self.bilstm = None
            classifier_in = feat_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, num_tags)
        self.crf = CRF(num_tags, batch_first=True) if use_crf else None

    def get_word_reps(self, input_ids, attention_mask, first_subword_idx):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx_expanded = first_subword_idx.unsqueeze(-1).expand(-1, -1, out.size(-1))
        return torch.gather(out, 1, idx_expanded)

    def forward(self, input_ids, attention_mask, first_subword_idx, word_mask, pos_ids, tag_ids=None):
        word_reps = self.get_word_reps(input_ids, attention_mask, first_subword_idx)
        pos_emb = self.pos_embed(pos_ids)
        combined = torch.cat([word_reps, pos_emb], dim=-1)

        if self.use_bilstm:
            feats, _ = self.bilstm(combined)
        else:
            feats = combined

        emissions = self.classifier(self.dropout(feats))

        if self.use_crf:
            if tag_ids is not None:
                return -self.crf(emissions, tag_ids, mask=word_mask, reduction="mean")
            return self.crf.decode(emissions, mask=word_mask)
        else:
            if tag_ids is not None:
                B, T, C = emissions.shape
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(emissions.view(B * T, C), tag_ids.view(B * T))
                mask_flat = word_mask.view(B * T).float()
                return (losses * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                pred_ids = emissions.argmax(dim=-1)
                out = []
                for i in range(pred_ids.size(0)):
                    length = int(word_mask[i].sum().item())
                    out.append(pred_ids[i, :length].tolist())
                return out

def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

# scoring functions, copied exactly from the original LoRA scripts
def strict_match_score(gold, pred):
    gold_set, pred_set = set(gold), set(pred)
    tp = len(gold_set & pred_set); fp = len(pred_set - gold_set); fn = len(gold_set - pred_set)
    return tp, fp, fn

def relaxed_match_score(gold, pred):
    gold = list(gold)
    used = [False] * len(gold)
    tp = fp = 0
    for (ps, pe, plab) in pred:
        matched = False
        for i, (gs, ge, glab) in enumerate(gold):
            if used[i]:
                continue
            if glab == plab and ps < ge and gs < pe:
                used[i] = True; matched = True; tp += 1; break
        if not matched:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn

def boundary_match_score(gold, pred):
    gold_set = {(s, e) for (s, e, _l) in gold}
    pred_set = {(s, e) for (s, e, _l) in pred}
    tp = len(gold_set & pred_set); fp = len(pred_set - gold_set); fn = len(gold_set - pred_set)
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1

def token_bioes_to_char_spans(tags, starts, ends):
    spans = []
    start_i, label = None, None
    for i, tag in enumerate(tags):
        if tag.startswith("S-"):
            spans.append((starts[i], ends[i], tag[2:])); start_i, label = None, None
        elif tag.startswith("B-"):
            start_i, label = i, tag[2:]
        elif tag.startswith("E-") and start_i is not None:
            spans.append((starts[start_i], ends[i], label)); start_i, label = None, None
        elif tag == "O":
            start_i, label = None, None
    return spans

def evaluate_on_docs(docs, model, tokenizer, tag2id, pos2id, id2tag, raw_by_id, collect_details=False):
    model.eval()
    strict_tp = strict_fp = strict_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    bound_tp = bound_fp = bound_fn = 0
    details = [] if collect_details else None

    ds = NERDataset(docs, tokenizer, tag2id, pos2id, CONFIG["max_words"])

    with torch.no_grad():
        for i, item in enumerate(docs):
            example = ds[i]
            batch = collate_fn([example])
            batch = {k: v.to(CONFIG["device"]) for k, v in batch.items()}

            preds = model(batch["input_ids"], batch["attention_mask"],
                           batch["first_subword_idx"], batch["word_mask"], batch["pos_ids"])
            pred_tag_ids = preds[0]
            length = min(len(pred_tag_ids), len(item["starts"]))
            pred_tags = [id2tag[t] for t in pred_tag_ids[:length]]
            pred_spans = token_bioes_to_char_spans(pred_tags, item["starts"][:length], item["ends"][:length])

            raw_item = raw_by_id[item["id"]]
            gold_spans = [(e["start"], e["end"], e["label"]) for e in raw_item["entities"]]

            tp, fp, fn = strict_match_score(gold_spans, pred_spans); strict_tp += tp; strict_fp += fp; strict_fn += fn
            tp, fp, fn = relaxed_match_score(gold_spans, pred_spans); relax_tp += tp; relax_fp += fp; relax_fn += fn
            tp, fp, fn = boundary_match_score(gold_spans, pred_spans); bound_tp += tp; bound_fp += fp; bound_fn += fn

            if collect_details:
                gold_set = set(gold_spans)
                pred_set = set(pred_spans)
                details.append({
                    "id": item["id"],
                    "text": raw_item["text"],
                    "gold_spans": [{"start": s, "end": e, "label": l, "text": raw_item["text"][s:e]} for s, e, l in gold_spans],
                    "pred_spans": [{"start": s, "end": e, "label": l, "text": raw_item["text"][s:e]} for s, e, l in pred_spans],
                    "missed_by_model_fn": [{"start": s, "end": e, "label": l, "text": raw_item["text"][s:e]}
                                            for s, e, l in (gold_set - pred_set)],
                    "false_positives_fp": [{"start": s, "end": e, "label": l, "text": raw_item["text"][s:e]}
                                            for s, e, l in (pred_set - gold_set)],
                    "num_tokens": len(item["words"]),
                })

    result = {
        "strict": dict(zip(["precision", "recall", "f1"], prf(strict_tp, strict_fp, strict_fn))),
        "relaxed": dict(zip(["precision", "recall", "f1"], prf(relax_tp, relax_fp, relax_fn))),
        "boundary": dict(zip(["precision", "recall", "f1"], prf(bound_tp, bound_fp, bound_fn))),
    }
    return (result, details) if collect_details else result

def result_path(variant_name, fold_idx):
    return os.path.join(RESULTS_DIR, f"{variant_name}_fold{fold_idx}.json")

def already_done(variant_name, fold_idx):
    path = result_path(variant_name, fold_idx)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return True, json.load(f)
    return False, None

def save_result(variant_name, fold_idx, result):
    path = result_path(variant_name, fold_idx)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  saved {path}")

def run_one_fold(variant_name, variant_cfg, fold_idx, train_idx, test_idx):
    print(f"\n{variant_name.upper()} fold {fold_idx}/{N_FOLDS - 1}")

    train_idx_list = list(train_idx)
    random.Random(SEED + fold_idx).shuffle(train_idx_list)
    n_dev = int(len(train_idx_list) * DEV_FRAC_WITHIN_TRAIN)
    dev_sub_idx = train_idx_list[:n_dev]
    train_sub_idx = train_idx_list[n_dev:]

    train_docs = [processed_data[i] for i in train_sub_idx]
    dev_docs = [processed_data[i] for i in dev_sub_idx]
    test_docs = [processed_data[i] for i in test_idx]

    train_ids = {d["id"] for d in train_docs}
    dev_ids = {d["id"] for d in dev_docs}
    test_ids = {d["id"] for d in test_docs}
    assert not (train_ids & dev_ids) and not (train_ids & test_ids) and not (dev_ids & test_ids), \
        f"Leakage detected in {variant_name} fold {fold_idx}."

    print(f"  Train: {len(train_docs)} | Dev: {len(dev_docs)} | Test: {len(test_docs)}")

    train_ds = NERDataset(train_docs, tokenizer, tag2id, pos2id, CONFIG["max_words"])
    dev_ds = NERDataset(dev_docs, tokenizer, tag2id, pos2id, CONFIG["max_words"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = NERModel(
        CONFIG["encoder_name"], len(tag2id), len(pos2id),
        CONFIG["lstm_hidden"], CONFIG["pos_dim"], CONFIG["dropout"],
        use_bilstm=variant_cfg["use_bilstm"], use_crf=variant_cfg["use_crf"],
    ).to(CONFIG["device"])

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": CONFIG["lr_encoder"]},
        {"params": head_params, "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(CONFIG["device"] == "cuda"))

    def eval_loss(loader):
        model.eval()
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, CONFIG["device"])
                loss = model(batch["input_ids"], batch["attention_mask"],
                             batch["first_subword_idx"], batch["word_mask"],
                             batch["pos_ids"], batch["tag_ids"])
                total += loss.item()
        return total / len(loader)

    best_dev_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, CONFIG["device"])
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(CONFIG["device"] == "cuda")):
                loss = model(batch["input_ids"], batch["attention_mask"],
                             batch["first_subword_idx"], batch["word_mask"],
                             batch["pos_ids"], batch["tag_ids"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        dv_loss = eval_loss(dev_loader)
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | dev_loss={dv_loss:.4f}")

        if dv_loss < best_dev_loss:
            best_dev_loss = dv_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to(CONFIG["device"])

    collect = (variant_name, fold_idx) == DETAIL_DUMP_TARGET
    if collect:
        test_result, details = evaluate_on_docs(test_docs, model, tokenizer, tag2id, pos2id, id2tag,
                                                  raw_by_id, collect_details=True)
        detail_path = os.path.join(RESULTS_DIR, f"DETAIL_{variant_name}_fold{fold_idx}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"  saved per-sentence detail dump to {detail_path}")
    else:
        test_result = evaluate_on_docs(test_docs, model, tokenizer, tag2id, pos2id, id2tag, raw_by_id)

    print(f"  {variant_name} fold {fold_idx} strict F1={test_result['strict']['f1']:.4f} "
          f"| relaxed F1={test_result['relaxed']['f1']:.4f} | boundary F1={test_result['boundary']['f1']:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "variant": variant_name, "fold": fold_idx,
        "n_train": len(train_docs), "n_dev": len(dev_docs), "n_test": len(test_docs),
        "result": test_result,
    }

# run all variants across all folds, resumable
all_results = {}
for variant_name, variant_cfg in ABLATION_VARIANTS.items():
    all_results[variant_name] = {}
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        done, prior = already_done(variant_name, fold_idx)
        if done:
            print(f"[resume] {variant_name} fold {fold_idx} already done, "
                  f"strict F1={prior['result']['strict']['f1']:.4f}. Skipping.")
            all_results[variant_name][fold_idx] = prior
            continue
        fold_result = run_one_fold(variant_name, variant_cfg, fold_idx, train_idx, test_idx)
        save_result(variant_name, fold_idx, fold_result)
        all_results[variant_name][fold_idx] = fold_result

# aggregate, mean plus std per variant
print("\nABLATION SUMMARY, English, 5 fold CV")
summary = {}
for variant_name in ABLATION_VARIANTS:
    strict_f1s = [all_results[variant_name][i]["result"]["strict"]["f1"] for i in range(N_FOLDS)]
    relaxed_f1s = [all_results[variant_name][i]["result"]["relaxed"]["f1"] for i in range(N_FOLDS)]
    boundary_f1s = [all_results[variant_name][i]["result"]["boundary"]["f1"] for i in range(N_FOLDS)]

    summary[variant_name] = {
        "strict_f1_mean": float(np.mean(strict_f1s)), "strict_f1_std": float(np.std(strict_f1s)),
        "relaxed_f1_mean": float(np.mean(relaxed_f1s)), "relaxed_f1_std": float(np.std(relaxed_f1s)),
        "boundary_f1_mean": float(np.mean(boundary_f1s)), "boundary_f1_std": float(np.std(boundary_f1s)),
        "per_fold_strict_f1": [round(x, 4) for x in strict_f1s],
    }

    print(f"\n{variant_name} (use_bilstm={ABLATION_VARIANTS[variant_name]['use_bilstm']}, "
          f"use_crf={ABLATION_VARIANTS[variant_name]['use_crf']}):")
    print(f"  Strict F1  : {np.mean(strict_f1s):.4f} +/- {np.std(strict_f1s):.4f}  per fold: {[round(x,4) for x in strict_f1s]}")
    print(f"  Relaxed F1 : {np.mean(relaxed_f1s):.4f} +/- {np.std(relaxed_f1s):.4f}")
    print(f"  Boundary F1: {np.mean(boundary_f1s):.4f} +/- {np.std(boundary_f1s):.4f}")

summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({"config": CONFIG, "variants": ABLATION_VARIANTS, "summary": summary,
                "fold_dataset_statistics": fold_stats}, f, ensure_ascii=False, indent=2)
print(f"\nSaved ablation summary to: {summary_path}")

# fold 1 diagnostic readout, uses the detail dump saved during the run above
detail_path = os.path.join(RESULTS_DIR, "DETAIL_full_fold1.json")
with open(detail_path, encoding="utf-8") as f:
    details = json.load(f)

total_fn = sum(len(d["missed_by_model_fn"]) for d in details)
total_fp = sum(len(d["false_positives_fp"]) for d in details)

fn_by_label = {}
fp_by_label = {}
for d in details:
    for e in d["missed_by_model_fn"]:
        fn_by_label[e["label"]] = fn_by_label.get(e["label"], 0) + 1
    for e in d["false_positives_fp"]:
        fp_by_label[e["label"]] = fp_by_label.get(e["label"], 0) + 1

print("\nFOLD 1 DIAGNOSTIC, full architecture")
print(f"Total sentences: {len(details)}")
print(f"Total missed entities, false negatives: {total_fn}")
print(f"Total false positive entities: {total_fp}")
print(f"\nMissed entities by label: {fn_by_label}")
print(f"False positives by label: {fp_by_label}")

print("\nWorst 5 sentences, most combined FN and FP")
ranked = sorted(details, key=lambda d: len(d["missed_by_model_fn"]) + len(d["false_positives_fp"]), reverse=True)
for d in ranked[:5]:
    print(f"\nid={d['id']} | tokens={d['num_tokens']} | text: {d['text'][:150]}")
    print(f"  Missed FN: {[e['text'] + '/' + e['label'] for e in d['missed_by_model_fn']]}")
    print(f"  False positives FP: {[e['text'] + '/' + e['label'] for e in d['false_positives_fp']]}")

print(f"\nFull per-sentence detail available at: {detail_path}")
