import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict

# install missing deps
def _ensure_installed():
    pkgs = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "accelerate": "accelerate",
        "pandas": "pandas",
        "sentencepiece": "sentencepiece",
        "huggingface_hub": "huggingface_hub",
        "sklearn": "scikit-learn",
    }
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[setup] installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _ensure_torchao_version(min_version="0.16.0"):
    # Colab's pre-installed torchao is too old for peft's dispatch check, blocks LoRA for every model
    needs_upgrade = False
    try:
        import torchao
        from packaging import version
        if version.parse(torchao.__version__) < version.parse(min_version):
            print(f"[setup] torchao {torchao.__version__} is older than the "
                  f"{min_version} peft requires -- upgrading.")
            needs_upgrade = True
    except ImportError:
        needs_upgrade = True

    if needs_upgrade:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                         f"torchao>={min_version}"], check=True)
        print("[setup] torchao upgraded. If LoRA attachment still fails with the same "
              "ImportError, the old version was already imported earlier in this Colab "
              "session -- use Runtime > Restart Session and rerun this cell once "
              "(the upgrade itself will persist, no need to reinstall).")


_ensure_installed()
_ensure_torchao_version()

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] CUDA available: {torch.cuda.is_available()}")

CANONICAL_LABELS = ["PERSON", "LOCATION", "ORGANIZATION", "DATE", "TIME", "NUMBER", "DESIGNATION"]
RANDOM_SEED = 42
TRAIN_FRAC, DEV_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
EPOCHS = 2  # loss was already flattening well before epoch 3

# edit to run a subset; resume support skips whatever's already done
MODELS_TO_RUN = ["bloomz_560m", "gemma_4_e2b", "bloomz_3b", "mt0_large"]

MODEL_CONFIGS = {
    "bloomz_560m": {
        "model_id": "bigscience/bloomz-560m", "arch": "causal_plain",
        "target_modules": ["query_key_value"], "quantize": True,
        "batch_size": 2, "grad_accum": 4,
    },
    "gemma_4_e2b": {
        "model_id": "google/gemma-4-E2B-it", "arch": "causal_chat",
        # regex targets the real nn.Linear inside Gemma4's custom wrapper, not the wrapper itself
        "target_modules": r".*(q_proj|k_proj|v_proj|o_proj)\.linear$",
        "quantize": False,
        "batch_size": 1, "grad_accum": 8,  # bf16, no quantization, more memory/step
    },
    "bloomz_3b": {
        "model_id": "bigscience/bloomz-3b", "arch": "causal_plain",
        "target_modules": ["query_key_value"], "quantize": True,
        "batch_size": 2, "grad_accum": 4,
    },
    "mt0_large": {
        "model_id": "bigscience/mt0-large", "arch": "seq2seq",
        "target_modules": ["q", "k", "v", "o"], "quantize": True,
        "batch_size": 2, "grad_accum": 4,
    },
}

# ---- data loading + gold spans ----

def load_gold(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_spans(sentence):
    return [(e["start"], e["end"], e["label"]) for e in sentence["entities"]]


def gold_json_target(sentence):
    by_label = defaultdict(list)
    for e in sentence["entities"]:
        by_label[e["label"]].append(e["text"])
    obj = {label: by_label.get(label, []) for label in CANONICAL_LABELS}
    return json.dumps(obj, ensure_ascii=False)


SYSTEM_PROMPT = (
    "You are a named entity recognition assistant. Given a sentence (which may be in Urdu), "
    "extract every named entity you find into a JSON object with exactly these keys: "
    "PERSON, LOCATION, ORGANIZATION, DATE, TIME, NUMBER, DESIGNATION. "
    "Each key's value must be a JSON array of the exact substrings from the sentence "
    "(verbatim, not paraphrased or translated) that belong to that entity type. "
    "If there are no entities of a given type, use an empty array. "
    "Respond with ONLY the JSON object -- no explanation, no markdown code fences."
)

# BLOOMZ has no chat template
PLAIN_PROMPT_TEMPLATE = SYSTEM_PROMPT + "\n\nSentence: {text}\n\nJSON:"
# mT0 has no chat template either, but follows direct instructions from its xP3 tuning
SEQ2SEQ_PROMPT_TEMPLATE = SYSTEM_PROMPT + "\n\nSentence: {text}\n\nJSON:"


def build_causal_prompt(tokenizer, text, use_chat_template):
    if use_chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sentence: {text}\n\nJSON:"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return PLAIN_PROMPT_TEMPLATE.format(text=text)


# ---- scoring ----

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


def parse_prediction(text, raw_output):
    # parses model JSON output into (start, end, label) spans matched back into the sentence
    spans = []
    try:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else json.loads(raw_output)
    except Exception:
        return spans

    cursor_by_label = defaultdict(int)
    for label in CANONICAL_LABELS:
        values = parsed.get(label, [])
        if isinstance(values, (str, int, float)):
            values = [str(values)]
        if not isinstance(values, list):
            continue
        for val in values:
            if not val or not isinstance(val, str):
                continue
            start_search = cursor_by_label[label]
            idx = text.find(val, start_search)
            if idx == -1:
                idx = text.find(val)
            if idx == -1:
                continue
            end = idx + len(val)
            spans.append((idx, end, label))
            cursor_by_label[label] = end
    return spans


def evaluate_causal(gold_data, indices, model, tokenizer, use_chat_template, max_new_tokens=250,
                     progress_every=20, progress_label="eval"):
    strict_tp = strict_fp = strict_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    bound_tp = bound_fp = bound_fn = 0

    t0 = time.time()
    n = len(indices)
    for i, idx in enumerate(indices):
        sent = gold_data[idx]
        text = sent["text"]
        gold = gold_spans(sent)

        prompt = build_causal_prompt(tokenizer, text, use_chat_template)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0][enc["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = parse_prediction(text, raw_output)

        tp, fp, fn = strict_match_score(gold, pred); strict_tp += tp; strict_fp += fp; strict_fn += fn
        tp, fp, fn = relaxed_match_score(gold, pred); relax_tp += tp; relax_fp += fp; relax_fn += fn
        tp, fp, fn = boundary_match_score(gold, pred); bound_tp += tp; bound_fp += fp; bound_fn += fn

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"    [{progress_label}] {i+1}/{n} sentences "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"~{elapsed/(i+1):.1f}s/sentence)")

    _, _, s_f1 = prf(strict_tp, strict_fp, strict_fn)
    _, _, r_f1 = prf(relax_tp, relax_fp, relax_fn)
    _, _, b_f1 = prf(bound_tp, bound_fp, bound_fn)
    return {"strict_f1": round(s_f1, 4), "relaxed_f1": round(r_f1, 4), "boundary_f1": round(b_f1, 4)}


def evaluate_seq2seq(gold_data, indices, model, tokenizer, max_new_tokens=250,
                      progress_every=20, progress_label="eval"):
    strict_tp = strict_fp = strict_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    bound_tp = bound_fp = bound_fn = 0

    t0 = time.time()
    n = len(indices)
    for i, idx in enumerate(indices):
        sent = gold_data[idx]
        text = sent["text"]
        gold = gold_spans(sent)

        prompt = SEQ2SEQ_PROMPT_TEMPLATE.format(text=text)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        raw_output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        pred = parse_prediction(text, raw_output)

        tp, fp, fn = strict_match_score(gold, pred); strict_tp += tp; strict_fp += fp; strict_fn += fn
        tp, fp, fn = relaxed_match_score(gold, pred); relax_tp += tp; relax_fp += fp; relax_fn += fn
        tp, fp, fn = boundary_match_score(gold, pred); bound_tp += tp; bound_fp += fp; bound_fn += fn

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"    [{progress_label}] {i+1}/{n} sentences "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"~{elapsed/(i+1):.1f}s/sentence)")

    _, _, s_f1 = prf(strict_tp, strict_fp, strict_fn)
    _, _, r_f1 = prf(relax_tp, relax_fp, relax_fn)
    _, _, b_f1 = prf(bound_tp, bound_fp, bound_fn)
    return {"strict_f1": round(s_f1, 4), "relaxed_f1": round(r_f1, 4), "boundary_f1": round(b_f1, 4)}


# ---- model loading (4-bit where applicable) + LoRA ----

def load_causal_model(model_id, hf_token, quantize=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    load_kwargs = dict(device_map="auto", token=hf_token, trust_remote_code=False)
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # trust_remote_code=False avoids stale custom repo code breaking generate()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        print(f"  [load] plain AutoModelForCausalLM failed ({exc}) -- "
              f"trying AutoModelForImageTextToText fallback.")
        import transformers
        AutoModelClass = getattr(transformers, "AutoModelForImageTextToText", None) or \
            getattr(transformers, "AutoModelForVision2Seq")
        model = AutoModelClass.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_seq2seq_model(model_id, hf_token):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", token=hf_token,
    )
    return model, tokenizer


def attach_lora(model, arch, target_modules):
    # fixes "element 0 of tensors does not require grad": frozen base model needs this to keep the graph connected
    model.enable_input_require_grads()
    model.config.use_cache = False

    task_type = TaskType.SEQ_2_SEQ_LM if arch == "seq2seq" else TaskType.CAUSAL_LM
    lora_config = LoraConfig(
        task_type=task_type, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---- fine-tuning datasets + training loops ----

class CausalSFTDataset(torch.utils.data.Dataset):
    def __init__(self, gold_data, indices, tokenizer, use_chat_template, max_length=1536):
        self.examples = []
        for idx in indices:
            sent = gold_data[idx]
            prompt = build_causal_prompt(tokenizer, sent["text"], use_chat_template)
            target = gold_json_target(sent) + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True,
                                    max_length=max_length)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False, truncation=True,
                                    max_length=max_length)["input_ids"]

            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def causal_collate_fn(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


class Seq2SeqSFTDataset(torch.utils.data.Dataset):
    def __init__(self, gold_data, indices, tokenizer, max_length=1024, max_target_length=512):
        self.examples = []
        for idx in indices:
            sent = gold_data[idx]
            prompt = SEQ2SEQ_PROMPT_TEMPLATE.format(text=sent["text"])
            target = gold_json_target(sent)

            input_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
            # text_target= replaces the removed as_target_tokenizer() context manager
            target_ids = tokenizer(text_target=target, truncation=True,
                                    max_length=max_target_length)["input_ids"]

            self.examples.append({"input_ids": input_ids, "labels": target_ids})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def seq2seq_collate_fn(batch, pad_token_id):
    max_in = max(len(b["input_ids"]) for b in batch)
    max_out = max(len(b["labels"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        in_pad = max_in - len(b["input_ids"])
        out_pad = max_out - len(b["labels"])
        input_ids.append(b["input_ids"] + [pad_token_id] * in_pad)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * in_pad)
        labels.append(b["labels"] + [-100] * out_pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def fine_tune(model, tokenizer, gold_data, train_indices, dev_indices, output_dir,
              arch, use_chat_template=True, epochs=2, batch_size=2, grad_accum=4):
    if arch == "seq2seq":
        train_ds = Seq2SeqSFTDataset(gold_data, train_indices, tokenizer)
        collate = lambda b: seq2seq_collate_fn(b, tokenizer.pad_token_id or 0)
    else:
        train_ds = CausalSFTDataset(gold_data, train_indices, tokenizer, use_chat_template)
        collate = lambda b: causal_collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
    trainer.train()
    return model


# ---- Drive save + resume ----

def _mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        results_dir = '/content/drive/MyDrive/urdu_split_finetune_results'
        os.makedirs(results_dir, exist_ok=True)
        print(f"[drive] Google Drive mounted. Results saved to: {results_dir}")
        return results_dir
    except Exception as exc:
        fallback_dir = "results_urdu_split_finetune"
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"[drive] Could not mount Google Drive ({exc}) -- using local {fallback_dir}/.")
        return fallback_dir


def result_path(results_dir, model_key):
    return os.path.join(results_dir, f"{model_key}_result.json")


def zeroshot_partial_path(results_dir, model_key):
    return os.path.join(results_dir, f"{model_key}_zeroshot_only.json")


def already_done(results_dir, model_key):
    path = result_path(results_dir, model_key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return True, json.load(f)
    return False, None


def zeroshot_already_done(results_dir, model_key):
    # separate checkpoint so a fine-tuning failure doesn't force redoing zero-shot too
    path = zeroshot_partial_path(results_dir, model_key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return True, json.load(f)
    return False, None


def save_zeroshot_partial(results_dir, model_key, zero_shot_scores):
    path = zeroshot_partial_path(results_dir, model_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(zero_shot_scores, f, indent=2)
    print(f"  [drive] saved partial (zero-shot only) {path}")


def save_result(results_dir, model_key, result):
    path = result_path(results_dir, model_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [drive] saved {path}")
    partial = zeroshot_partial_path(results_dir, model_key)
    if os.path.exists(partial):
        os.remove(partial)


# ---- run ----

RESULTS_DIR = _mount_drive()

HF_TOKEN = None
try:
    entered = input("Enter your Hugging Face token (required for Gemma 4 -- gated model): ").strip()
    if entered:
        from huggingface_hub import login
        login(token=entered, add_to_git_credential=False)
        HF_TOKEN = entered
        print("[auth] Logged in to Hugging Face Hub.")
    else:
        print("[auth] No token provided -- gemma_4_e2b will likely fail (gated model).")
except Exception:
    print("[auth] Could not prompt for a token in this environment.")

try:
    from google.colab import files
    print("\nUpload your Urdu gold-standard JSON file (uner_eval.json / merged_eval_set.json format):")
    uploaded = files.upload()
    INPUT_PATH = next(iter(uploaded))
    print(f"[upload] received {INPUT_PATH} ({len(uploaded[INPUT_PATH])} bytes)\n")
except ImportError:
    INPUT_PATH = input("Not running in Colab -- enter the path to your gold JSON file: ").strip()

gold_data = load_gold(INPUT_PATH)
print(f"[data] loaded {len(gold_data)} sentences from {INPUT_PATH}")

all_indices = list(range(len(gold_data)))
train_idx, temp_idx = train_test_split(all_indices, test_size=(DEV_FRAC + TEST_FRAC),
                                        random_state=RANDOM_SEED)
dev_idx, test_idx = train_test_split(temp_idx, test_size=TEST_FRAC / (DEV_FRAC + TEST_FRAC),
                                      random_state=RANDOM_SEED)
print(f"[data] split: {len(train_idx)} train / {len(dev_idx)} dev / {len(test_idx)} test")

for model_key in MODELS_TO_RUN:
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'#' * 70}\nMODEL: {model_key} ({cfg['model_id']}, arch={cfg['arch']})\n{'#' * 70}")

    done, prior = already_done(RESULTS_DIR, model_key)
    if done:
        print(f"  [resume] {model_key} already done -- "
              f"zero_shot strict_f1={prior['zero_shot']['strict_f1']}, "
              f"fine_tuned strict_f1={prior['fine_tuned']['strict_f1']}. Skipping.")
        continue

    try:
        arch = cfg["arch"]
        use_chat_template = (arch == "causal_chat")

        if arch == "seq2seq":
            model, tokenizer = load_seq2seq_model(cfg["model_id"], HF_TOKEN)
        else:
            model, tokenizer = load_causal_model(cfg["model_id"], HF_TOKEN, quantize=cfg["quantize"])

        zs_done, zero_shot_scores = zeroshot_already_done(RESULTS_DIR, model_key)
        if zs_done:
            print(f"  [resume] zero-shot already computed for {model_key} -- "
                  f"skipping recompute: {zero_shot_scores}")
        else:
            print("  [eval] zero-shot on test set...")
            if arch == "seq2seq":
                zero_shot_scores = evaluate_seq2seq(gold_data, test_idx, model, tokenizer,
                                                     progress_label=f"{model_key} zero-shot")
            else:
                zero_shot_scores = evaluate_causal(gold_data, test_idx, model, tokenizer, use_chat_template,
                                                    progress_label=f"{model_key} zero-shot")
            print(f"  [eval] zero-shot: {zero_shot_scores}")
            save_zeroshot_partial(RESULTS_DIR, model_key, zero_shot_scores)

        print("  [train] LoRA fine-tuning on train split...")
        model = attach_lora(model, arch, cfg["target_modules"])
        model = fine_tune(model, tokenizer, gold_data, train_idx, dev_idx,
                           output_dir=f"_ft_{model_key}", arch=arch,
                           use_chat_template=use_chat_template,
                           epochs=EPOCHS,
                           batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"])

        print("  [eval] fine-tuned on test set...")
        model.eval()
        model.config.use_cache = True
        if arch == "seq2seq":
            fine_tuned_scores = evaluate_seq2seq(gold_data, test_idx, model, tokenizer,
                                                  progress_label=f"{model_key} fine-tuned")
        else:
            fine_tuned_scores = evaluate_causal(gold_data, test_idx, model, tokenizer, use_chat_template,
                                                 progress_label=f"{model_key} fine-tuned")
        print(f"  [eval] fine-tuned: {fine_tuned_scores}")

        result = {
            "model": model_key, "model_id": cfg["model_id"], "arch": arch,
            "n_train": len(train_idx), "n_dev": len(dev_idx), "n_test": len(test_idx),
            "zero_shot": zero_shot_scores, "fine_tuned": fine_tuned_scores,
        }
        save_result(RESULTS_DIR, model_key, result)

    except Exception as exc:
        print(f"  [error] {model_key} failed: {exc}")
        traceback.print_exc()
        print(f"  [note] zero-shot (if completed) is saved separately and won't be recomputed.")
    finally:
        try:
            del model, tokenizer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(f"_ft_{model_key}", ignore_errors=True)

print("\n[done] All requested models finished (or already done from a previous run).")
rows = []
for model_key in MODELS_TO_RUN:
    done, result = already_done(RESULTS_DIR, model_key)
    if done:
        rows.append({
            "model": model_key,
            "zs_strict": result["zero_shot"]["strict_f1"],
            "zs_relaxed": result["zero_shot"]["relaxed_f1"],
            "zs_boundary": result["zero_shot"]["boundary_f1"],
            "ft_strict": result["fine_tuned"]["strict_f1"],
            "ft_relaxed": result["fine_tuned"]["relaxed_f1"],
            "ft_boundary": result["fine_tuned"]["boundary_f1"],
        })
if rows:
    print(pd.DataFrame(rows).to_string(index=False))
print(f"\n[done] All results saved in Google Drive at: {RESULTS_DIR}")
