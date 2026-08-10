import json, random, os, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF

# install: pip install -q transformers accelerate stanza pytorch-crf

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
print("Device:", CONFIG["device"])

# mount drive for saving results
from google.colab import drive
drive.mount('/content/drive')

RESULTS_DIR = "/content/drive/MyDrive/urdu_ner_structured/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Results will be saved to:", RESULTS_DIR)

# upload gold json at runtime
from google.colab import files
print("Upload your uner_eval.json file:")
uploaded = files.upload()
DATA_PATH = list(uploaded.keys())[0]
print("Loaded file:", DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Total documents: {len(raw_data)}")
all_labels = sorted({e["label"] for d in raw_data for e in d["entities"]})
print("Entity types found:", all_labels)

# keep a lookup to the original raw entities, true gold, exact char offsets
raw_by_id = {item["id"]: item for item in raw_data}

# stanza tokenization + BIOES alignment
import stanza
stanza.download('ur', verbose=False)
nlp = stanza.Pipeline('ur', processors='tokenize,pos', verbose=False,
                       use_gpu=(CONFIG["device"] == "cuda"))

def get_tokens_with_offsets(text):
    doc = nlp(text)
    tokens = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            upos = tok.words[0].upos if tok.words else "X"
            tokens.append({
                "text": tok.text,
                "start": tok.start_char,
                "end": tok.end_char,
                "pos": upos
            })
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
            continue
        tokens = get_tokens_with_offsets(text)
        if not tokens:
            skipped += 1
            continue
        tags = entities_to_bioes(tokens, item["entities"])
        processed.append({
            "id": item["id"],
            "words": [t["text"] for t in tokens],
            "pos": [t["pos"] for t in tokens],
            "tags": tags,
            "starts": [t["start"] for t in tokens],
            "ends": [t["end"] for t in tokens],
        })
    print(f"Processed {len(processed)} docs, skipped {skipped}")
    return processed

processed_data = process_dataset(raw_data)
print(processed_data[0])

# build vocabularies and split 70/15/15, matches the original LoRA script's split method
all_tags = sorted({t for d in processed_data for t in d["tags"]})
tag2id = {t: i for i, t in enumerate(all_tags)}
id2tag = {i: t for t, i in tag2id.items()}

all_pos = sorted({p for d in processed_data for p in d["pos"]})
pos2id = {p: i + 1 for i, p in enumerate(all_pos)}
pos2id["<PAD>"] = 0

print(f"Num tags: {len(tag2id)}")
print(f"Num POS tags: {len(pos2id)}")

from sklearn.model_selection import train_test_split

DEV_FRAC, TEST_FRAC = 0.15, 0.15
all_positions = list(range(len(processed_data)))

train_pos, temp_pos = train_test_split(all_positions, test_size=(DEV_FRAC + TEST_FRAC),
                                        random_state=SEED)
dev_pos, test_pos = train_test_split(temp_pos, test_size=TEST_FRAC / (DEV_FRAC + TEST_FRAC),
                                      random_state=SEED)

train_data = [processed_data[i] for i in train_pos]
dev_data = [processed_data[i] for i in dev_pos]
test_data = [processed_data[i] for i in test_pos]

print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

# leakage and duplicate checks, run before training
train_ids = {d["id"] for d in train_data}
dev_ids = {d["id"] for d in dev_data}
test_ids = {d["id"] for d in test_data}

print("Split leakage check, should all be empty sets")
print("train n dev :", train_ids & dev_ids)
print("train n test:", train_ids & test_ids)
print("dev n test  :", dev_ids & test_ids)
assert not (train_ids & dev_ids) and not (train_ids & test_ids) and not (dev_ids & test_ids), \
    "Leakage detected between splits, do not proceed until fixed."

train_texts = {raw_by_id[d["id"]]["text"] for d in train_data}
dev_texts = {raw_by_id[d["id"]]["text"] for d in dev_data}
test_texts = {raw_by_id[d["id"]]["text"] for d in test_data}

dup_train_test = train_texts & test_texts
dup_train_dev = train_texts & dev_texts

print("Exact duplicate sentence check")
print(f"Identical text in train and test: {len(dup_train_test)} out of {len(test_texts)} test sentences")
print(f"Identical text in train and dev : {len(dup_train_dev)} out of {len(dev_texts)} dev sentences")
if dup_train_test:
    print("Warning, duplicate sentences appear in both train and test. Examples:", list(dup_train_test)[:3])

# dataset and collate, subword to word pooling
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
        seen = set()
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                first_subword_idx.append(i)
                seen.add(wid)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
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

train_ds = NERDataset(train_data, tokenizer, tag2id, pos2id, CONFIG["max_words"])
dev_ds = NERDataset(dev_data, tokenizer, tag2id, pos2id, CONFIG["max_words"])
test_ds = NERDataset(test_data, tokenizer, tag2id, pos2id, CONFIG["max_words"])

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

# model: XLM-R plus POS embedding plus BiLSTM plus CRF
class EncoderBiLSTMCRF(nn.Module):
    def __init__(self, encoder_name, num_tags, num_pos, lstm_hidden, pos_dim, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        self.pos_embed = nn.Embedding(num_pos, pos_dim, padding_idx=0)
        self.bilstm = nn.LSTM(enc_dim + pos_dim, lstm_hidden, batch_first=True,
                               bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def get_word_reps(self, input_ids, attention_mask, first_subword_idx):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx_expanded = first_subword_idx.unsqueeze(-1).expand(-1, -1, out.size(-1))
        word_reps = torch.gather(out, 1, idx_expanded)
        return word_reps

    def forward(self, input_ids, attention_mask, first_subword_idx, word_mask, pos_ids, tag_ids=None):
        word_reps = self.get_word_reps(input_ids, attention_mask, first_subword_idx)
        pos_emb = self.pos_embed(pos_ids)
        combined = torch.cat([word_reps, pos_emb], dim=-1)
        lstm_out, _ = self.bilstm(combined)
        emissions = self.classifier(self.dropout(lstm_out))

        if tag_ids is not None:
            loss = -self.crf(emissions, tag_ids, mask=word_mask, reduction="mean")
            return loss
        else:
            preds = self.crf.decode(emissions, mask=word_mask)
            return preds

model = EncoderBiLSTMCRF(
    CONFIG["encoder_name"], len(tag2id), len(pos2id),
    CONFIG["lstm_hidden"], CONFIG["pos_dim"], CONFIG["dropout"]
).to(CONFIG["device"])

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# optimizer, scheduler, training loop with early stopping and mixed precision
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

def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def run_eval_loss(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, CONFIG["device"])
            loss = model(batch["input_ids"], batch["attention_mask"],
                         batch["first_subword_idx"], batch["word_mask"],
                         batch["pos_ids"], batch["tag_ids"])
            total_loss += loss.item()
    return total_loss / len(loader)

best_dev_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_loss = 0
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
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    dev_loss = run_eval_loss(dev_loader)
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping.")
            break

model.load_state_dict(best_state)
model.to(CONFIG["device"])
print("Loaded best model, lowest dev loss.")

# scoring functions, copied exactly from the original LoRA fine-tuning script
def strict_match_score(gold, pred):
    gold_set, pred_set = set(gold), set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
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
                used[i] = True
                matched = True
                tp += 1
                break
        if not matched:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn

def boundary_match_score(gold, pred):
    gold_set = {(s, e) for (s, e, _l) in gold}
    pred_set = {(s, e) for (s, e, _l) in pred}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
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
            spans.append((starts[i], ends[i], tag[2:]))
            start_i, label = None, None
        elif tag.startswith("B-"):
            start_i, label = i, tag[2:]
        elif tag.startswith("E-") and start_i is not None:
            spans.append((starts[start_i], ends[i], label))
            start_i, label = None, None
        elif tag == "O":
            start_i, label = None, None
    return spans

def evaluate_test_set(test_data, model, tokenizer, tag2id, pos2id, id2tag, raw_by_id):
    model.eval()
    strict_tp = strict_fp = strict_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    bound_tp = bound_fp = bound_fn = 0

    ds = NERDataset(test_data, tokenizer, tag2id, pos2id, CONFIG["max_words"])

    with torch.no_grad():
        for i, item in enumerate(test_data):
            example = ds[i]
            batch = collate_fn([example])
            batch = {k: v.to(CONFIG["device"]) for k, v in batch.items()}

            preds = model(batch["input_ids"], batch["attention_mask"],
                           batch["first_subword_idx"], batch["word_mask"],
                           batch["pos_ids"])
            pred_tag_ids = preds[0]
            length = min(len(pred_tag_ids), len(item["starts"]))
            pred_tags = [id2tag[t] for t in pred_tag_ids[:length]]

            pred_spans = token_bioes_to_char_spans(pred_tags, item["starts"][:length], item["ends"][:length])

            raw_item = raw_by_id[item["id"]]
            gold_spans = [(e["start"], e["end"], e["label"]) for e in raw_item["entities"]]

            tp, fp, fn = strict_match_score(gold_spans, pred_spans)
            strict_tp += tp; strict_fp += fp; strict_fn += fn

            tp, fp, fn = relaxed_match_score(gold_spans, pred_spans)
            relax_tp += tp; relax_fp += fp; relax_fn += fn

            tp, fp, fn = boundary_match_score(gold_spans, pred_spans)
            bound_tp += tp; bound_fp += fp; bound_fn += fn

    return {
        "strict": dict(zip(["precision", "recall", "f1"], prf(strict_tp, strict_fp, strict_fn))),
        "relaxed": dict(zip(["precision", "recall", "f1"], prf(relax_tp, relax_fp, relax_fn))),
        "boundary": dict(zip(["precision", "recall", "f1"], prf(bound_tp, bound_fp, bound_fn))),
    }

test_results = evaluate_test_set(test_data, model, tokenizer, tag2id, pos2id, id2tag, raw_by_id)
print(json.dumps(test_results, indent=2))

# save everything to drive
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RESULTS_DIR, f"xlmr_bilstm_crf_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({
        "config": CONFIG,
        "test_results": test_results,
        "train_size": len(train_data),
        "dev_size": len(dev_data),
        "test_size": len(test_data),
        "tag_set": all_tags,
    }, f, ensure_ascii=False, indent=2)

torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

with open(os.path.join(run_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump({"tag2id": tag2id, "pos2id": pos2id}, f, ensure_ascii=False, indent=2)

print(f"Saved everything to: {run_dir}")
print(json.dumps(test_results, indent=2))

# seed stability check, run separately after the above.
# reruns the whole pipeline under a few different seeds and reports mean plus std
# strict/relaxed/boundary F1, same reporting style as the English LoRA results.
SEEDS_TO_TEST = [42, 123, 7]

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_splits(seed):
    all_positions = list(range(len(processed_data)))
    train_pos, temp_pos = train_test_split(all_positions, test_size=(DEV_FRAC + TEST_FRAC),
                                            random_state=seed)
    dev_pos, test_pos = train_test_split(temp_pos, test_size=TEST_FRAC / (DEV_FRAC + TEST_FRAC),
                                          random_state=seed)
    tr = [processed_data[i] for i in train_pos]
    dv = [processed_data[i] for i in dev_pos]
    te = [processed_data[i] for i in test_pos]
    return tr, dv, te

def check_leakage(tr, dv, te):
    tr_ids, dv_ids, te_ids = {d["id"] for d in tr}, {d["id"] for d in dv}, {d["id"] for d in te}
    assert not (tr_ids & dv_ids) and not (tr_ids & te_ids) and not (dv_ids & te_ids), \
        "Leakage detected, aborting this seed run."
    tr_txt = {raw_by_id[d["id"]]["text"] for d in tr}
    te_txt = {raw_by_id[d["id"]]["text"] for d in te}
    dup = tr_txt & te_txt
    if dup:
        print(f"    Warning, {len(dup)} duplicate sentences between train/test for this seed.")

def train_and_eval_one_seed(seed):
    print(f"\nSEED {seed}")
    set_all_seeds(seed)

    tr, dv, te = build_splits(seed)
    check_leakage(tr, dv, te)
    print(f"  Train: {len(tr)} | Dev: {len(dv)} | Test: {len(te)}")

    train_ds_s = NERDataset(tr, tokenizer, tag2id, pos2id, CONFIG["max_words"])
    dev_ds_s = NERDataset(dv, tokenizer, tag2id, pos2id, CONFIG["max_words"])
    train_loader_s = DataLoader(train_ds_s, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    dev_loader_s = DataLoader(dev_ds_s, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    model_s = EncoderBiLSTMCRF(
        CONFIG["encoder_name"], len(tag2id), len(pos2id),
        CONFIG["lstm_hidden"], CONFIG["pos_dim"], CONFIG["dropout"]
    ).to(CONFIG["device"])

    encoder_params_s = list(model_s.encoder.parameters())
    head_params_s = [p for n, p in model_s.named_parameters() if not n.startswith("encoder.")]
    optimizer_s = torch.optim.AdamW([
        {"params": encoder_params_s, "lr": CONFIG["lr_encoder"]},
        {"params": head_params_s, "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    total_steps_s = len(train_loader_s) * CONFIG["epochs"]
    scheduler_s = get_linear_schedule_with_warmup(optimizer_s, num_warmup_steps=int(0.1 * total_steps_s),
                                                   num_training_steps=total_steps_s)
    scaler_s = torch.amp.GradScaler('cuda', enabled=(CONFIG["device"] == "cuda"))

    def eval_loss(loader):
        model_s.eval()
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, CONFIG["device"])
                loss = model_s(batch["input_ids"], batch["attention_mask"],
                                batch["first_subword_idx"], batch["word_mask"],
                                batch["pos_ids"], batch["tag_ids"])
                total += loss.item()
        return total / len(loader)

    best_dev_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        model_s.train()
        for batch in train_loader_s:
            batch = move_batch(batch, CONFIG["device"])
            optimizer_s.zero_grad()
            with torch.amp.autocast('cuda', enabled=(CONFIG["device"] == "cuda")):
                loss = model_s(batch["input_ids"], batch["attention_mask"],
                                batch["first_subword_idx"], batch["word_mask"],
                                batch["pos_ids"], batch["tag_ids"])
            scaler_s.scale(loss).backward()
            scaler_s.unscale_(optimizer_s)
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            scaler_s.step(optimizer_s)
            scaler_s.update()
            scheduler_s.step()

        dv_loss = eval_loss(dev_loader_s)
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | dev_loss={dv_loss:.4f}")

        if dv_loss < best_dev_loss:
            best_dev_loss = dv_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model_s.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print("  Early stopping.")
                break

    model_s.load_state_dict(best_state)
    model_s.to(CONFIG["device"])

    result = evaluate_test_set(te, model_s, tokenizer, tag2id, pos2id, id2tag, raw_by_id)
    print(f"  strict F1={result['strict']['f1']:.4f} | relaxed F1={result['relaxed']['f1']:.4f} "
          f"| boundary F1={result['boundary']['f1']:.4f}")

    del model_s, optimizer_s, scheduler_s
    torch.cuda.empty_cache()
    gc_import.collect()

    return result

import gc as gc_import

all_seed_results = {}
for seed in SEEDS_TO_TEST:
    all_seed_results[seed] = train_and_eval_one_seed(seed)

strict_f1s = [all_seed_results[s]["strict"]["f1"] for s in SEEDS_TO_TEST]
relaxed_f1s = [all_seed_results[s]["relaxed"]["f1"] for s in SEEDS_TO_TEST]
boundary_f1s = [all_seed_results[s]["boundary"]["f1"] for s in SEEDS_TO_TEST]

print(f"\nSEED STABILITY SUMMARY, {len(SEEDS_TO_TEST)} seeds: {SEEDS_TO_TEST}")
print(f"Strict F1  : {np.mean(strict_f1s):.4f} +/- {np.std(strict_f1s):.4f}  per seed: {[round(x,4) for x in strict_f1s]}")
print(f"Relaxed F1 : {np.mean(relaxed_f1s):.4f} +/- {np.std(relaxed_f1s):.4f}  per seed: {[round(x,4) for x in relaxed_f1s]}")
print(f"Boundary F1: {np.mean(boundary_f1s):.4f} +/- {np.std(boundary_f1s):.4f}  per seed: {[round(x,4) for x in boundary_f1s]}")

seed_summary_path = os.path.join(RESULTS_DIR, f"seed_stability_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(seed_summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "seeds": SEEDS_TO_TEST,
        "per_seed_results": all_seed_results,
        "summary": {
            "strict_f1_mean": float(np.mean(strict_f1s)), "strict_f1_std": float(np.std(strict_f1s)),
            "relaxed_f1_mean": float(np.mean(relaxed_f1s)), "relaxed_f1_std": float(np.std(relaxed_f1s)),
            "boundary_f1_mean": float(np.mean(boundary_f1s)), "boundary_f1_std": float(np.std(boundary_f1s)),
        }
    }, f, ensure_ascii=False, indent=2)
print(f"\nSaved seed stability summary to: {seed_summary_path}")
