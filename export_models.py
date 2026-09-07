"""
Run this INSIDE your Colab notebook (paste as a new cell, after Scenario A has been run in
Part A, so `lr_model`/`vectorizer` from Scenario A exist) to export the trained Logistic
Regression model for use in the PoC app. RoBERTa checkpoints are already saved automatically
by the notebook (Section 3.2.9) — just copy the `roberta_scenario_A` folder from Drive into
this app's `models/roberta_final/` folder.

Usage in Colab:
    # Re-fit LR on the full ISOT training set (the PoC's "production" model — matches
    # Scenario A, the in-distribution ISOT classifier used in Table 3.5)
    lr_model, vectorizer, _ = train_logistic_regression(isot_train)

    import pickle, os
    os.makedirs(f'{DRIVE_ROOT}/poc_models', exist_ok=True)
    with open(f'{DRIVE_ROOT}/poc_models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open(f'{DRIVE_ROOT}/poc_models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print('Saved lr_model.pkl and tfidf_vectorizer.pkl to Drive/fake_news_thesis/poc_models/')

Then download both files from Drive/fake_news_thesis/poc_models/ and place them in this
app's models/ folder (alongside roberta_final/), matching the structure:

    poc/
      app.py
      models/
        lr_model.pkl
        tfidf_vectorizer.pkl
        roberta_final/
          config.json
          model.safetensors
          tokenizer.json
          ... (rest of the saved RoBERTa checkpoint files)
"""
