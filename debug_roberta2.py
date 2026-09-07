"""
Diagnostic script — run this in the SAME environment where models/roberta_final/ lives
(e.g. your Streamlit app's folder, or Colab if that's where the checkpoint is).

    python debug_roberta.py

It checks three things, in order:
  1. Does the checkpoint folder actually contain fine-tuned weights (not just config/tokenizer)?
  2. What does the model's config say (num_labels, id2label)?
  3. Does the model produce DIFFERENT logits for clearly different inputs, or does it collapse
     to (near-)identical output regardless of what you feed it?
"""

import os
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer as RobertaTokenizerFast

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ROBERTA_PATH = os.path.join(MODELS_DIR, "roberta_final")


def clean_text(text: str) -> str:
    import re
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Step 1: what's actually in the checkpoint folder? ───────────────────────────────
print("=" * 70)
print("STEP 1 — checkpoint folder contents")
print("=" * 70)
if not os.path.isdir(ROBERTA_PATH):
    raise SystemExit(f"❌ {ROBERTA_PATH} does not exist — nothing to debug.")

for fname in sorted(os.listdir(ROBERTA_PATH)):
    fpath = os.path.join(ROBERTA_PATH, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {fname:35s} {size_mb:10.2f} MB")

weight_files = [f for f in os.listdir(ROBERTA_PATH)
                 if f.endswith((".bin", ".safetensors"))]
if not weight_files:
    print("\n❌ NO WEIGHT FILE FOUND (no .bin or .safetensors) — this is your bug.")
    print("   The classifier head was never saved/copied. Re-copy the checkpoint from")
    print("   the Drive path used in the notebook (roberta_scenario_<X>/) and make sure")
    print("   BOTH the model weights AND tokenizer files come along.")
    raise SystemExit(1)
else:
    total_mb = sum(os.path.getsize(os.path.join(ROBERTA_PATH, f)) for f in weight_files) / (1024 * 1024)
    if total_mb < 400:
        print(f"\n⚠️  Weight file(s) total {total_mb:.1f} MB — a fine-tuned roberta-base")
        print("   classification checkpoint should be ~475-500 MB. This looks suspiciously")
        print("   small/truncated (possibly an interrupted Drive sync/copy).")
    else:
        print(f"\n✓ Weight file(s) total {total_mb:.1f} MB — looks like a complete roberta-base checkpoint.")

# ── Step 2: load and inspect config ──────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 2 — model config")
print("=" * 70)
model = RobertaForSequenceClassification.from_pretrained(ROBERTA_PATH)
tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_PATH)
model.eval()
print(f"  num_labels: {model.config.num_labels}")
print(f"  id2label:   {model.config.id2label}")

# ── Step 3: does it actually discriminate between different inputs? ─────────────────
print()
print("=" * 70)
print("STEP 3 — logits on clearly different inputs")
print("=" * 70)

test_cases = {
    "ISOT-style REAL (Reuters dateline)":
        "washington reuters u s president says trade talks will continue "
        "next week after both sides agreed to a new round of negotiations",
    "Obviously fabricated / sensational":
        "aliens land in new york city and demand to speak with the president "
        "shocking video footage reveals the truth they don t want you to know",
    "Neutral factual-sounding statement":
        "the local city council voted on tuesday to approve funding for a "
        "new public library branch in the downtown area",
    "Empty-ish / junk input":
        "asdkj aslkdj alskdj",
}

results = []
for label, raw_text in test_cases.items():
    cleaned = clean_text(raw_text)
    inputs = tokenizer(cleaned, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)
    pred = int(torch.argmax(probs))
    results.append((label, logits.tolist(), probs.tolist(), pred))
    print(f"\n  [{label}]")
    print(f"    raw logits : {[round(x, 4) for x in logits.tolist()]}")
    print(f"    probs      : {[round(x, 4) for x in probs.tolist()]}")
    print(f"    prediction : {'REAL' if pred == 1 else 'FAKE'}")

# ── Verdict ───────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
all_logits = [r[1] for r in results]
# crude check: how much do logits vary across wildly different inputs?
import statistics
class0_vals = [l[0] for l in all_logits]
class1_vals = [l[1] for l in all_logits]
spread0 = max(class0_vals) - min(class0_vals)
spread1 = max(class1_vals) - min(class1_vals)
all_same_pred = len(set(r[3] for r in results)) == 1

print(f"  logit spread (class 0): {spread0:.4f}")
print(f"  logit spread (class 1): {spread1:.4f}")
if all_same_pred and spread0 < 0.5 and spread1 < 0.5:
    print("\n❌ COLLAPSED MODEL: logits barely move across wildly different inputs, and every")
    print("   input gets the same prediction. The model is NOT discriminating on content —")
    print("   this points to a checkpoint/loading problem (see Step 1), not a dataset-bias")
    print("   issue. The fine-tuned classifier head likely wasn't loaded correctly.")
elif all_same_pred:
    print("\n⚠️  Same prediction across all test cases, but logits DO shift somewhat with")
    print("   input. The model is responding to content but may be biased/undertrained")
    print("   toward one class — check training class balance and epoch count for whichever")
    print("   scenario this checkpoint came from.")
else:
    print("\n✓ Predictions differ across inputs — the model IS discriminating on content.")
    print("   Your earlier 'always FAKE' results were likely specific to the particular")
    print("   articles you tried, not a systemic collapse. Worth testing with a wider")
    print("   variety of real/fake ISOT test-set rows to check calibration.")
