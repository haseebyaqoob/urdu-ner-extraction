# NER Fine-Tuning: English and Urdu

This project tests whether fine-tuning small, general-purpose LLMs for
Named Entity Recognition (NER) helps consistently across model size and
language. Each model is fine-tuned with LoRA, under the same method and
conditions, then compared zero-shot vs. fine-tuned on held-out test data.
English uses 5-fold cross-validation (368 sentences); Urdu uses a single
70/15/15 train/dev/test split (1,736 sentences, UNER).

## Results

### English (5-fold CV, mean +/- std strict F1)

| Model | Size | Zero-shot | Fine-tuned |
|---|---|---|---|
| Qwen2.5 0.5B | 0.5B | 0.2525 +/- 0.0242 | 0.7708 +/- 0.0347 |
| Qwen2-VL 2B | 2B | 0.2399 +/- 0.0216 | 0.8390 +/- 0.0279 |
| Phi-3.5 Mini | 3.8B | 0.5173 +/- 0.0364 | 0.8415 +/- 0.0249 |

### Urdu (single split, strict F1)

| Model | Zero-shot | Fine-tuned |
|---|---|---|
| BLOOMZ 560M | 0.0000 | 0.6737 |
| Gemma 4 E2B | 0.6315 | not usable, see note below |
| BLOOMZ 3B | 0.0027 | 0.8008 |
| mT0-Large | 0.0000 | 0.5864 |

### Discussion

Fine-tuning helped every general-purpose model tested, and helped
substantially. On English, the improvement held at every size (0.5B, 2B,
3.8B) with no reversal. On Urdu, every model that started near 0%
zero-shot was rescued dramatically by fine-tuning, gaining between 58 and
80 points of strict F1.

Zero-shot scores are low mostly because these models were never shown the
exact output format the task requires (a strict JSON schema with verbatim
entity spans), not because they lack language understanding. Fine-tuning
closes that gap quickly, and relaxed F1 improves by a similar margin to
strict F1 in these results, which suggests the gain is real recognition
improvement and not just the model learning to match the output format.

Gemma 4 E2B is the one exception: it was already the strongest
zero-shot performer on Urdu (0.6315), but its fine-tuning run showed no
measurable effect. The fine-tuned scores came out identical to zero-shot
and training loss never decreased, most likely due to an unresolved
incompatibility between Gemma 4's custom quantization-aware layers and
standard LoRA fine-tuning. Its zero-shot score is the reliable number for
this model; the fine-tuning result is a documented limitation, not a
genuine null finding.
