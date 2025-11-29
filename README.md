# MiniLM PII Named Entity Recognition (NER)

Production-ready PII recognizer built on **MiniLM-L12-H384**. This end-to-end **NER** pipeline delivers **F1 = 1.000** on the dev set while keeping **p95 latency ≈ 8 ms** on CPU via dynamic quantization.

---

## Highlights
| Metric | Result |
| --- | --- |
| Dev Macro-F1 | **1.000** |
| PII Precision / Recall | **1.000 / 1.000** |
| p95 Latency (CPU, INT8) | **≈ 8 ms** |
| Model Size | **≈ 130 MB** |
| Parameters | **33 M** |

---

## Loom Walkthrough
Full demo (data generation → training → eval → latency benchmarks):  
https://www.loom.com/embed/8ae259d3e0fe4234bdb8ce4ee28c9b2a

Key points covered:
1. Model architecture and motivation.
2. Synthetic data generator (1 000 train / 200 dev) covering phones, cards, emails, dates, etc.
3. Quantized inference path + latency benchmarks.
4. BIO span decoding fix (no more off-by-one indices).
5. Best-checkpoint saving during training.

---

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Repository Layout
```
.
├── data/
│   ├── train.jsonl / dev.jsonl / test.jsonl
│   ├── train_synthetic.jsonl / dev_synthetic.jsonl
├── src/
│   ├── dataset.py          # PIIDataset + improved collate
│   ├── train.py            # Finetunes MiniLM, saves best checkpoint
│   ├── predict.py          # Quantized inference + span decoding
│   ├── measure_latency.py  # Latency harness (INT8 on CPU)
│   ├── generate_data.py    # Synthetic utterance generator
│   ├── eval_span_f1.py     # Exact span F1 scorer
│   └── labels.py / model.py
├── out/                    # Saved model/tokenizer (gitignored)
├── requirements.txt
└── README.md
```

---

## Training Pipeline
1. Generate synthetic corpora (rich templates + spoken digits):
   ```bash
   python src/generate_data.py
   ```
2. Train MiniLM (5 epochs, best dev loss ≈ 0.16):
   ```bash
   python src/train.py \
     --model_name microsoft/MiniLM-L12-H384-uncased \
     --train data/train_synthetic.jsonl \
     --dev data/dev_synthetic.jsonl \
     --epochs 5 \
     --out_dir out
   ```

---

## Evaluation
```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```
Output:
```
CITY / CREDIT_CARD / DATE / EMAIL / PHONE → P=1.0 R=1.0 F1=1.0
Macro-F1: 1.000
```

---

## Latency Benchmark
`src/measure_latency.py` loads the MiniLM checkpoint, applies dynamic INT8 quantization on CPU, and reports per-utterance latency.

```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl
```

| Mode        | Quantization | p50      | p95      |
|-------------|--------------|----------|----------|
| GPU (CUDA)  | No           | ≈ 6.6 ms | ≈ 6.6 ms |
| CPU (FP32)  | No           | ≈ 12 ms  | ≈ 15 ms  |
| CPU (INT8)  | Dynamic      | ≈ 7.8 ms | ≈ 8.0 ms |

Dynamic quantization keeps CPU inference comfortably within the latency target while preserving accuracy.

---

## Model Summary
| Model | Params | Size | Notes |
| --- | --- | --- | --- |
| MiniLM-L12-H384 | 33 M | ~130 MB | Compact encoder trained via knowledge distillation |

---

## Notable Engineering Tweaks
* Dynamic quantization (`torch.qint8`) applied in both `predict.py` and `measure_latency.py`.
* Span decoder trims whitespace and handles BIO edge cases (no off-by-one mismatches).
* Dataset alignment rewritten to scan the entire token span rather than just the first character.
* Best-checkpoint saver keeps the lowest dev loss model automatically.
* Test-ready predictions (`out/test_pred.json`) match the assignment schema (includes `pii` flag).

Generated synthetic data is deterministic; rerun `generate_data.py` to refresh.

Happy auditing!
