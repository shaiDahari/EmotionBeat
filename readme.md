```markdown
# EmotionBeat: Multi-Emotion Regression on Song Lyrics

**Repository:** https://github.com/shaiDahari/EmotionBeat

---

## 📖 Project Overview

EmotionBeat implements an end-to-end pipeline to predict six continuous emotion intensities—Joy, Sadness, Anger, Fear, Surprise and Tenderness—from a single concatenated text input:

```

Title: … | Genre: … | Artist: … | Lyrics: …

````

We compare four approaches:
- **Baseline** (mean predictor)
- **BERTForMultiRegression** (fine-tuned `bert-base-uncased`)
- **RoBERTaForMultiRegression** (fine-tuned `roberta-base`)
- **Zero-Shot LLM** (Azure Grok_3 API)

Key features:
- Data cleaning, formatting and tokenization
- Baseline, supervised fine-tuning and zero-shot inference
- Hyperparameter grid-search with cross-validation
- Evaluation on held-out test set (MSE & MAE)
- All code contained in one reusable notebook

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/shaiDahari/EmotionBeat.git
   cd EmotionBeat
````

2. **Install dependencies**

   ```bash
   pip install torch transformers pandas numpy scikit-learn openpyxl
   ```

3. **Run the notebook**

   * Open `SER_Complete_Pipeline.ipynb` in Jupyter Notebook or Google Colab
   * Execute all cells to reproduce preprocessing, model training (baseline, BERT, RoBERTa), zero-shot inference and evaluation

4. **Inspect outputs**

   * Prediction CSVs (`baseline_preds.csv`, `bert_predictions.csv`, etc.)
   * Summary tables: `table1_df.csv` (overall MSE/MAE) & `table2_df.csv` (per-emotion MSE/MAE)
   * Present slides in `NLP_FINAL_PTT.pptx`

---

## 📂 Repository Structure

```
EmotionBeat/
├── 500_song_tagging.xlsx            # Raw data: metadata, cleaned lyrics, MOS labels (0–2)
├── SER_Complete_Pipeline.ipynb      # Full pipeline: preprocessing → modeling → evaluation
├── NLP_FINAL_PTT.pptx               # Final presentation deck
├── Final Nlp Presentation.pdf         # Assignment guidelines (slides 34–end)
├── baseline_preds.csv               # Baseline predictions
├── baseline_truth.csv               # Baseline ground truth
├── bert_predictions.csv             # BERT predictions
├── bert_truth.csv                   # BERT ground truth
├── roberta_predictions.csv          # RoBERTa predictions
├── roberta_truth.csv                # RoBERTa ground truth
├── zero_shot_predictions.csv        # Zero-shot predictions
├── zero_shot_truth.csv              # Zero-shot ground truth
├── table1_df.csv                    # Overall MSE/MAE summary
├── table2_df.csv                    # Per-emotion MSE/MAE breakdown
└── README.md                        # This file
```

---

## 🧩 Methodology

1. **Data Preparation**

   * Clean and merge metadata + lyrics into one input string
   * Split into train/val/test (70/15/15, `random_state=42`)
   * Tokenize with the appropriate transformer tokenizer

2. **Baseline**

   * Compute and predict the training-set mean emotion vector

3. **Transformer Fine-Tuning**

   * Fine-tune BERT and RoBERTa using AdamW + MSE loss
   * Grid-search hyperparameters (learning rate, batch size, epochs, weight decay) with cross-validation
   * Early stopping on validation MSE

4. **Zero-Shot Inference**

   * Send a structured prompt to Azure Grok\_3 API
   * Parse the returned JSON of six emotion scores

5. **Evaluation**

   * Compute overall & per-emotion MSE/MAE on the test set
   * Compare all four approaches

---

## 📊 Results

* **Overall metrics** (`table1_df.csv`): MSE & MAE for each model
* **Per-emotion metrics** (`table2_df.csv`): breakdown across Joy, Sadness, Anger, Fear, Surprise, Tenderness

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request for bug fixes or enhancements.


