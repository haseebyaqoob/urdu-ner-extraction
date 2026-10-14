# NER Fine-Tuning: English and Urdu

This project tests whether fine-tuning small, general-purpose LLMs for
Named Entity Recognition (NER) helps consistently across model size and
language, and compares that approach against a specialized discriminative
architecture (XLM-R + BiLSTM + CRF) trained from scratch on the same data.

English uses 5-fold cross-validation (368 sentences); Urdu uses a
70/15/15 train/dev/test split (1,736 sentences, UNER).

## Results

### English (5-fold CV, mean +/- std strict F1)

| Approach | Strict F1 |
|---|---|
| Zero-shot, best (Phi-3.5 Mini) | 0.5173 +/- 0.0364 |
| LoRA fine-tuned, best (Phi-3.5 Mini) | 0.8415 +/- 0.0249 |
| XLM-R + BiLSTM + CRF (full) | **0.8721 +/- 0.0171** |
| XLM-R + BiLSTM only (no CRF) | 0.8703 +/- 0.0108 |
| XLM-R + CRF only (no BiLSTM) | 0.8551 +/- 0.0207 |

### Urdu (strict F1)

| Approach | Strict F1 |
|---|---|
| Zero-shot, best (Gemma 4 E2B) | 0.6315 |
| LoRA fine-tuned, best (BLOOMZ 3B) | 0.8008 |
| XLM-R + BiLSTM + CRF (3 independent runs, different splits/seeds) | 0.8800, 0.8810, 0.8716; mean 0.8775 +/- 0.0042 |

## Discussion

**Fine-tuning helps, but architecture matters more than fine-tuning method.**
The structured discriminative model (XLM-R + BiLSTM + CRF, fully
fine-tuned) beats the best LoRA-tuned generative LLM on both languages,
despite using far fewer parameters (280M vs 3-3.8B for the best
LoRA-tuned models, roughly 11-14x fewer, and about 7x fewer than the
roughly 2B-parameter models tested). The gap is largely mechanical: the
generative approach must produce a verbatim JSON string that is
string-matched back into the source text, so correct entities are lost to
formatting or paraphrasing noise. The discriminative model tags real
tokens at real positions and cannot make that class of error.

**Ablation: BiLSTM matters, CRF barely does (English only).** The
ablation was run on the English 5-fold setup only. Removing CRF cost ~0.2
points of strict F1 (arguably noise). Removing BiLSTM cost ~1.7 points and
increased variance. With 368 sentences and fold standard deviations of
roughly 1-2 points, the BiLSTM effect is suggestive rather than
conclusive. This is the opposite emphasis from some published Urdu NER
architectures, where CRF is reported as the larger contributor; worth
flagging as a dataset/domain-specific finding rather than a general rule.

**Fold 1 (English) is the consistent low point across all three
variants** (0.84 full / 0.85 no-CRF / 0.82 no-BiLSTM), but per-fold
dataset statistics (sentence length, entity density) do not explain it:
fold 1 is not the longest or densest fold. Manual inspection of the
worst-scoring sentences shows the errors are boundary-convention
disagreements, not missed entities: parenthetical titles ("Director
General (DG)"), possessives ("Rosberg's"), and multi-token numeric/time
spans ("one minute 37.583 seconds") where the model's span choice differs
from the annotator's by a few characters. Relaxed F1 on fold 1 (0.9346)
is much closer to the other folds than strict F1 (0.8409), consistent
with this being a boundary-convention issue rather than a recognition
failure.

**Gemma 4 E2B (Urdu, LoRA)** was the strongest zero-shot performer
(0.6315) but showed no measurable effect from fine-tuning: training loss
never decreased, most likely due to an incompatibility between Gemma 4's
custom quantization-aware layers and standard LoRA. This cause was not
verified. Documented as a limitation, not a null finding.

## Repo structure

- `english/`: English NER
  - `english_ner_finetune_kfold.py`: LoRA fine-tuning, 5-fold CV, general-purpose LLMs
  - `structured_english_bilstm_crf.py`: XLM-R + BiLSTM + CRF with the ablation variants (full, no CRF, no BiLSTM) across 5 folds
- `urdu/`: Urdu NER
  - `urdu_ner_finetune_split.py`: LoRA fine-tuning, single split, general-purpose LLMs
  - `structured_urdu_bilstm_crf.py`: XLM-R + BiLSTM + CRF
- `dataset/`: ZUNERA dataset cleaning and preprocessing (`cleaning_zunera.py`, `preprocessing_zunera.py`) and the processed file `zunera.txt`
- `papers/`: reference papers
