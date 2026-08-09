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


_ensure_installed()

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] CUDA available: {torch.cuda.is_available()}")

CANONICAL_LABELS = ["PERSON", "LOCATION", "ORGANIZATION", "DATE", "TIME", "NUMBER", "DESIGNATION"]
N_FOLDS = 5
RANDOM_SEED = 42

MODELS_TO_RUN = ["qwen2_5_0_5b", "qwen2_vl_2b", "phi_3_5_mini"]

MODEL_IDS = {
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B-Instruct",
    "phi_3_5_mini": "microsoft/Phi-3.5-mini-instruct",
}

#  data loading + gold spans 

def load_gold(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def gold_spans(sentence):
    return [(e["start"], e["end"], e["label"]) for e in sentence["entities"]]


def gold_json_target(sentence):
    # builds the target JSON string used as the fine-tuning label
    by_label = defaultdict(list)
    for e in sentence["entities"]:
        by_label[e["label"]].append(e["text"])
    obj = {label: by_label.get(label, []) for label in CANONICAL_LABELS}
    return json.dumps(obj, ensure_ascii=False)


SYSTEM_PROMPT = (
    "You are a named entity recognition assistant. Given a sentence, "
    "extract every named entity you find into a JSON object with exactly these keys: "
    "PERSON, LOCATION, ORGANIZATION, DATE, TIME, NUMBER, DESIGNATION. "
    "Each key's value must be a JSON array of the exact substrings from the sentence "
    "(verbatim, not paraphrased) that belong to that entity type. "
    "If there are no entities of a given type, use an empty array. "
    "Respond with ONLY the JSON object -- no explanation, no markdown code fences."
)


def build_prompt(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sentence: {text}\n\nJSON:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def evaluate_on_indices(gold_data, indices, model, tokenizer, max_new_tokens=600):
    strict_tp = strict_fp = strict_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    bound_tp = bound_fp = bound_fn = 0

    for idx in indices:
        sent = gold_data[idx]
        text = sent["text"]
        gold = gold_spans(sent)

        prompt = build_prompt(tokenizer, text)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0][enc["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = parse_prediction(text, raw_output)

        tp, fp, fn = strict_match_score(gold, pred)
        strict_tp += tp; strict_fp += fp; strict_fn += fn
        tp, fp, fn = relaxed_match_score(gold, pred)
        relax_tp += tp; relax_fp += fp; relax_fn += fn
        tp, fp, fn = boundary_match_score(gold, pred)
        bound_tp += tp; bound_fp += fp; bound_fn += fn

    _, _, s_f1 = prf(strict_tp, strict_fp, strict_fn)
    _, _, r_f1 = prf(relax_tp, relax_fp, relax_fn)
    _, _, b_f1 = prf(bound_tp, bound_fp, bound_fn)
    return {"strict_f1": round(s_f1, 4), "relaxed_f1": round(r_f1, 4), "boundary_f1": round(b_f1, 4)}


# ---- model loading (4-bit) + LoRA ----

def load_base_model(model_id, hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # trust_remote_code=False: avoids stale custom repo code breaking generate()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            token=hf_token, trust_remote_code=False,
        )
    except Exception as exc:
        # Qwen2-VL is vision-language, falls back to the image-text class (text-only here)
        print(f"  [load] plain AutoModelForCausalLM failed ({exc}) -- "
              f"trying AutoModelForImageTextToText (Qwen2-VL-style fallback).")
        import transformers
        AutoModelClass = getattr(transformers, "AutoModelForImageTextToText", None) or \
            getattr(transformers, "AutoModelForVision2Seq")
        model = AutoModelClass.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            token=hf_token, trust_remote_code=False,
        )
    return model, tokenizer


def attach_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


#  fine-tuning dataset + training loop 

class NERSFTDataset(torch.utils.data.Dataset):
    # prompt tokens masked out of the loss (labels=-100) so only the JSON completion is learned
    def __init__(self, gold_data, indices, tokenizer, max_length=1536):
        self.examples = []
        for idx in indices:
            sent = gold_data[idx]
            prompt = build_prompt(tokenizer, sent["text"])
            target = gold_json_target(sent) + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True,
                                    max_length=max_length)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False, truncation=True,
                                    max_length=max_length)["input_ids"]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate_fn(batch, pad_token_id):
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


def fine_tune(model, tokenizer, gold_data, train_indices, output_dir, epochs=3):
    dataset = NERSFTDataset(gold_data, train_indices, tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    trainer.train()
    return model


# ---- Drive save + resume ----

def _mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        results_dir = '/content/drive/MyDrive/english_kfold_finetune_results'
        os.makedirs(results_dir, exist_ok=True)
        print(f"[drive] Google Drive mounted. Results saved to: {results_dir}")
        return results_dir
    except Exception as exc:
        fallback_dir = "results_english_kfold"
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"[drive] Could not mount Google Drive ({exc}) -- using local {fallback_dir}/.")
        return fallback_dir


def fold_result_path(results_dir, model_key, fold_idx):
    return os.path.join(results_dir, f"{model_key}_fold{fold_idx}.json")


def fold_already_done(results_dir, model_key, fold_idx):
    path = fold_result_path(results_dir, model_key, fold_idx)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return True, json.load(f)
    return False, None


def save_fold_result(results_dir, model_key, fold_idx, result):
    path = fold_result_path(results_dir, model_key, fold_idx)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [drive] saved {path}")


def summarize(results_dir, model_key, n_folds):
    zero_shot = {"strict_f1": [], "relaxed_f1": [], "boundary_f1": []}
    fine_tuned = {"strict_f1": [], "relaxed_f1": [], "boundary_f1": []}
    for fold_idx in range(n_folds):
        done, result = fold_already_done(results_dir, model_key, fold_idx)
        if not done:
            continue
        for metric in zero_shot:
            zero_shot[metric].append(result["zero_shot"][metric])
            fine_tuned[metric].append(result["fine_tuned"][metric])

    def mean_std(vals):
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return round(m, 4), round(var ** 0.5, 4)

    summary = {"model": model_key, "n_folds_completed": len(zero_shot["strict_f1"])}
    for phase_name, phase in [("zero_shot", zero_shot), ("fine_tuned", fine_tuned)]:
        for metric, vals in phase.items():
            m, s = mean_std(vals)
            summary[f"{phase_name}_{metric}_mean"] = m
            summary[f"{phase_name}_{metric}_std"] = s
    return summary


#  run 

RESULTS_DIR = _mount_drive()

HF_TOKEN = None
try:
    entered = input("Enter your Hugging Face token (required for Phi-3.5/Qwen weights): ").strip()
    if entered:
        from huggingface_hub import login
        login(token=entered, add_to_git_credential=False)
        HF_TOKEN = entered
        print("[auth] Logged in to Hugging Face Hub.")
    else:
        print("[auth] No token provided -- gated/rate-limited models may fail.")
except Exception:
    print("[auth] Could not prompt for a token in this environment.")

try:
    from google.colab import files
    print("\nUpload your gold-standard JSON file (merged_eval_set.json):")
    uploaded = files.upload()
    INPUT_PATH = next(iter(uploaded))
    print(f"[upload] received {INPUT_PATH} ({len(uploaded[INPUT_PATH])} bytes)\n")
except ImportError:
    INPUT_PATH = input("Not running in Colab -- enter the path to your gold JSON file: ").strip()

gold_data = load_gold(INPUT_PATH)
print(f"[data] loaded {len(gold_data)} sentences from {INPUT_PATH}")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
all_indices = list(range(len(gold_data)))
folds = list(kf.split(all_indices))
print(f"[data] built {N_FOLDS} folds "
      f"(~{len(folds[0][0])} train / ~{len(folds[0][1])} held-out per fold)")

for model_key in MODELS_TO_RUN:
    model_id = MODEL_IDS[model_key]
    print(f"\n{'#' * 70}\nMODEL: {model_key} ({model_id})\n{'#' * 70}")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        done, prior = fold_already_done(RESULTS_DIR, model_key, fold_idx)
        if done:
            print(f"  [resume] fold {fold_idx} already done -- "
                  f"zero_shot strict_f1={prior['zero_shot']['strict_f1']}, "
                  f"fine_tuned strict_f1={prior['fine_tuned']['strict_f1']}. Skipping.")
            continue

        print(f"\n  --- fold {fold_idx}/{N_FOLDS - 1} "
              f"({len(train_idx)} train, {len(test_idx)} held-out) ---")
        try:
            model, tokenizer = load_base_model(model_id, HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("  [eval] zero-shot on held-out fold...")
            zero_shot_scores = evaluate_on_indices(gold_data, test_idx, model, tokenizer)
            print(f"  [eval] zero-shot: {zero_shot_scores}")

            print("  [train] LoRA fine-tuning on remaining folds...")
            model = attach_lora(model)
            model = fine_tune(model, tokenizer, gold_data, train_idx,
                               output_dir=f"_ft_{model_key}_fold{fold_idx}")

            print("  [eval] fine-tuned on held-out fold...")
            model.eval()
            fine_tuned_scores = evaluate_on_indices(gold_data, test_idx, model, tokenizer)
            print(f"  [eval] fine-tuned: {fine_tuned_scores}")

            result = {
                "model": model_key, "fold": fold_idx,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "zero_shot": zero_shot_scores, "fine_tuned": fine_tuned_scores,
            }
            save_fold_result(RESULTS_DIR, model_key, fold_idx, result)

        except Exception as exc:
            print(f"  [error] fold {fold_idx} for {model_key} failed: {exc}")
            traceback.print_exc()
        finally:
            try:
                del model, tokenizer
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(f"_ft_{model_key}_fold{fold_idx}", ignore_errors=True)

    # mean +/- std summary across whatever folds are done so far
    summary = summarize(RESULTS_DIR, model_key, N_FOLDS)
    print(f"\n  [summary] {model_key}: {json.dumps(summary, indent=2)}")
    with open(os.path.join(RESULTS_DIR, f"{model_key}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

print("\n[done] All requested models finished (or already done from a previous run).")
print("[copy-paste summary table]")
rows = []
for model_key in MODELS_TO_RUN:
    path = os.path.join(RESULTS_DIR, f"{model_key}_summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
if rows:
    print(pd.DataFrame(rows).to_string(index=False))
print(f"\n[done] All results saved in Google Drive at: {RESULTS_DIR}")
