# Fake News Detection — Proof-of-Concept

A Streamlit web app demonstrating all three model families from the thesis
(Logistic Regression, RoBERTa, and a zero-shot LLM) in one interface, per
Chapter 3 Section 3.2.8.

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs immediately for the LLM model (just paste a Groq API key in the
sidebar — get one free at console.groq.com/keys). Logistic Regression and
RoBERTa will show a "model files not found" message until you add the
trained model artifacts — see below.

## Adding your trained models

1. **Logistic Regression** — see `export_models.py` for the exact code to run
   in your Colab notebook. It saves `lr_model.pkl` and `tfidf_vectorizer.pkl`.
   Download both and place them in `models/`.

2. **RoBERTa** — your notebook already saves a checkpoint to Google Drive
   after every scenario (`Drive/fake_news_thesis/checkpoints/roberta_scenario_A/`).
   Download that folder and rename it to `models/roberta_final/`.

Final folder structure:

```
poc/
  app.py
  requirements.txt
  models/
    lr_model.pkl
    tfidf_vectorizer.pkl
    roberta_final/
      config.json
      model.safetensors
      tokenizer.json
      ... (rest of the checkpoint)
```

## Deploying (Streamlit Cloud)

1. Push this folder to a GitHub repository (model files can be large — consider
   Git LFS for `roberta_final/`, or host RoBERTa on the HuggingFace Hub instead
   and point `load_roberta_model()` in `app.py` at that repo name).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub
   account, and deploy pointing at `app.py`.
3. In the app's "Secrets" settings, add:
   ```
   GROQ_API_KEY = "your-key-here"
   ```
   so visitors don't need to paste a key themselves.

## What each model shows

| Model | Confidence shown? | Notes |
|---|---|---|
| Logistic Regression | Yes (predicted class probability) | Sub-millisecond inference |
| RoBERTa | Yes (predicted class probability) | Slower, more accurate on in-distribution data |
| LLM (gpt-oss-120b) | No — zero-shot text generation only | Includes Groq API network latency |
