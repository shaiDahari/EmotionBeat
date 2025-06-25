
# 🎧 EmotionBeat: Multi-Emotion Regression on Song Lyrics

Welcome to **EmotionBeat**, a deep learning NLP project that predicts **multi-dimensional emotional intensities** from song lyrics. Instead of simple tags, our system produces **six continuous emotion scores**—enabling mood-based music exploration like never before.

[🔗 Project Report (PDF)](https://github.com/shaiDahari/EmotionBeat/blob/main/Docs/NLP%20FINAL%20PTT.pdf) • [🧠 View Notebook](link-to-notebook) • [📊 See Results](#results-summary)

---

## 🔍 Problem Statement

Music isn’t just about genre—**it’s about how it makes us feel**. Unfortunately, current music metadata lacks **fine-grained emotional tagging**, limiting mood-based discovery.

**Goal:**  
Given a song's title, genre, artist, and lyrics, predict a vector of **six continuous emotion scores** (range `[0.0 – 2.0]`) for:
> Joy ∣ Sadness ∣ Anger ∣ Fear ∣ Surprise ∣ Tenderness

---

## 📦 Dataset

- 📂 **Source:** [Kaggle: Spotify Most Popular Songs](https://www.kaggle.com)
- 🎼 **Instances:** 497 songs (after filtering)
- 🧪 **Split:** Train/Val/Test → 70% / 15% / 15%
- 💾 **Fields:**
  - Title, Genre, Artist, Cleaned Lyrics
  - Emotion Scores (MOS aggregated from 6 annotators)

### 🧠 Labeling via Mean Opinion Scores (MOS)

Each song was rated by multiple annotators on a scale from **0.0 (none)** to **2.0 (high intensity)** per emotion. Final labels are the average across raters.

---

## 🧪 Exploratory Data Analysis (EDA)

- 🎯 **Dominant Emotions:** Joy, Tenderness
- 😨 **Least Represented:** Fear
- 🔗 **Interesting Correlations:**
  - Joy ↔ Sadness: -0.30
  - Anger ↔ Fear: +0.25
- 🎶 **Genre Trends:**
  - EDM → high Joy & Surprise
  - Rock → elevated Anger & Fear

---

## 🧰 Models & Pipeline

### 1. **Baseline**
- Predicts training-set mean vector for all songs.
- MSE ≈ 0.1511

### 2. **Transformer Fine-Tuning**
#### 🔧 `BERTForMultiRegression`
- Model: `bert-base-uncased`
- Custom regression head: 256-dim layer + ReLU + Dropout → 6 outputs
- Best Params: `lr=3e-5`, `batch=4`, `epochs=5`, `weight_decay=0.05`

#### 🔧 `RoBERTaForMultiRegression`
- Model: `roberta-base`
- Same architecture as above
- Best Params: `lr=1e-5`, `batch=8`, `epochs=8`, `weight_decay=0.01`

### 3. **Zero-Shot LLM**
- Model: `Azure Grok_3`
- Prompt-based inference only
- No training involved

---

## 📈 Training Configuration

- Framework: HuggingFace Transformers
- Optimizer: AdamW
- Loss: MSELoss
- Hardware: Google Colab Pro, GPU L4
- Search Space: 144 combinations (LR, epochs, batch size, weight decay)

---

## 📊 Results Summary

| Model       | MSE ↓     | MAE ↓     | Notes                            |
|-------------|-----------|-----------|----------------------------------|
| **BERT**    | **0.1507**| 0.3331    | Best overall                     |
| RoBERTa     | 0.1511    | **0.3334**| Best on Surprise & Tenderness    |
| Baseline    | 0.1511    | 0.3335    | Mean predictor                   |
| Zero-Shot   | 0.3872    | 0.5096    | Struggles with subtle lyrics     |

### 🎯 Per-Emotion Highlights:
- **BERT**: Strongest on Anger & Fear
- **RoBERTa**: Best at Surprise & Tenderness
- **Zero-Shot**: Underperforms on metaphorical text

---

## 🧱 Project Structure

```bash
.
├── Data Set
│   └── 500_song_tagging.xlsx
├── Docs
│   ├── EmotionBeat.pdf
│   └── EmotionBeat.pptx
├── Notebook
│   └── SER_Complete_Pipeline.ipynb
├── OutPuts
│   ├── baseline_preds.csv
│   ├── baseline_truth.csv
│   ├── bert_predictions.csv
│   ├── bert_truth.csv
│   ├── roberta_predictions.csv
│   ├── roberta_truth.csv
│   ├── zero_shot_predictions.csv
│   └── zero_shot_truth.csv
├── Results
│   └── table_1.png       # Overall MSE/MAE
│   └── table_2.png       # Per-emotion MSE/MAE
└── README.md
```

---

## 🛣️ Future Work

- 📈 **Scale Data**: Expand to 5K–10K labeled tracks
- 🧪 **Better Splits**: Use stratified K-Fold CV
- 🎛️ **Hyperparams**: Try dropout, GELU, larger hidden layers
- 🔍 **Interpretability**: Visualize attention weights on lyrics

---

## 📚 Citation

If you use this code or data in your research, please cite:

> EmotionBeat: Multi-Emotion Regression on Song Lyrics  
> [https://github.com/shaiDahari/EmotionBeat](https://github.com/shaiDahari/EmotionBeat)

---

## 🤝 Contributors

- Shai Dahari  
- Adane Abaye
- Mazal Lemlem
- Naor Matsliah

---
