# 📈 Yes Bank Stock Price Prediction

A supervised machine learning project to predict the **monthly closing price** of Yes Bank shares using historical OHLC stock data from July 2005 to November 2020.

---

## 📌 Problem Statement

Stock market price prediction is a challenging task due to the dynamic nature of financial markets. Yes Bank has been one of India's most volatile banking stocks — rising from ₹13 in 2005 to ₹400 in 2018, followed by a sharp decline due to aggressive lending practices, rising NPAs, and RBI regulatory interventions.

This project builds a machine learning model to predict Yes Bank's closing price while preserving the real-world volatility present in the data — treating the price crash as a genuine market event, not an outlier.

---

## 🎯 Objective

- Analyze historical stock price trends of Yes Bank (2005–2020)
- Identify key features influencing the monthly closing price
- Build and compare multiple regression models
- Use SHAP analysis to explain model predictions
- Select the best-performing model based on business-relevant metrics

---

## 📂 Dataset

- **Source:** Kaggle
- **Period:** July 2005 – November 2020
- **Size:** 185 rows × 5 columns
- **Frequency:** Monthly stock prices
- **Target Variable:** `Close`

### Features

| Feature | Description |
|---------|-------------|
| `Open`  | Monthly opening price (₹) |
| `High`  | Monthly highest price (₹) |
| `Low`   | Monthly lowest price (₹) |
| `Month` | Extracted from Date |
| `Year`  | Extracted from Date |

> `Date` column was parsed and Month/Year were extracted as separate features — quarterly financial results and yearly comparisons impact share price significantly.

---

## 🔍 EDA Highlights

- Stock price rose from ~₹13 (2005) to ~₹400 (Aug 2018) — then crashed sharply
- Price distribution is **right-skewed** — most values in ₹15–₹100 range
- **Strong positive correlation (0.98–1.00)** between Open, High, Low, and Close
- Outliers retained — price crash was caused by real business events (NPAs, RBI action), not data errors
- Month and Year show no significant seasonal pattern with price

---

## ⚙️ Preprocessing

- No missing values or duplicates found
- Month and Year extracted from Date column
- No categorical encoding needed
- `StandardScaler` applied before Linear Regression training
- 80:20 train-test split with `random_state=42`

---

## 🤖 Models Trained

### 1. Linear Regression (Final Model ✅)
- StandardScaler applied
- Ridge Regression with GridSearchCV (alpha tuning)
- Best Alpha: 0.001 | Best CV Score: 0.9946

### 2. Random Forest Regressor
- Default + GridSearchCV tuning
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

### 3. XGBoost Regressor
- Default + GridSearchCV tuning
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

---

## 📊 Results

| Model | R² Score | RMSE (₹) | MAE (₹) | MSE |
|-------|----------|-----------|---------|-----|
| **Linear Regression** | **99.07%** | **9.17** | **5.89** | **84.02** |
| Random Forest | 97.83% | 14.01 | 8.68 | 196.38 |
| RF + GridSearchCV | 97.90% | 13.77 | 9.08 | 189.60 |
| XGBoost | 97.53% | 14.95 | 9.93 | 223.37 |
| XGB + GridSearchCV | 97.52% | 14.96 | 9.72 | 223.82 |

---

## 🏆 Final Model — Linear Regression

**Linear Regression outperformed all ensemble models** across every metric:

- R² ~1.2% higher than best ensemble model
- RMSE ~4.6 points lower
- MAE ~3 points lower

**Why Linear Regression won:**
- Yes Bank's closing price has a **strong linear relationship** with Open, High, Low features
- Adding ensemble complexity introduced noise, not signal
- Fully interpretable — coefficients directly explain feature impact
- No overfitting — GridSearchCV on ensemble models showed no significant improvement
- Production-ready — lightweight, no hyperparameter tuning needed

---

## 🔍 SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to explain Linear Regression predictions.

| Feature | Mean |SHAP| | Interpretation |
|---------|-------------|----------------|
| **Low** | **~70** | Most dominant feature |
| High | ~43 | Second most influential |
| Open | ~38 | Sets price direction |
| Month | ~1 | Negligible contribution |
| Year | ~0 | Negligible contribution |

**Key Finding:** `Low` price is the single strongest predictor of closing price — the month's low acts as the primary anchor for where the stock closes. Temporal features (Month, Year) contribute near-zero SHAP values, confirming price levels drive closing price — not seasonality.

---

## 📉 Evaluation Metrics — Business Impact

| Metric | Value | Business Meaning |
|--------|-------|-----------------|
| R² Score | 99.07% | Model explains 99% of price variance — highly reliable |
| RMSE | ₹9.17 | Large errors minimized — critical for trading decisions |
| MAE | ₹5.90 | Average prediction error of only ₹5.90 per month |
| MSE | 84.02 | Training loss kept low — model penalizes large errors |

---

## ⚠️ Limitations

- OHLC features are highly correlated — model learns price relationships, not true market dynamics
- External factors not included — news sentiment, RBI policy, macroeconomic indicators
- Monthly data only — not suitable for intraday or weekly predictions
- Specific to Yes Bank — not generalized across other stocks

---

## 🛠️ Libraries Used

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
```

---

## 📁 Project Structure

```
├── Yes_Bank_Stock_Prediction.ipynb    # Main notebook
├── data/
│   └── data_YesBank_StockPrices.csv  # Dataset
├── yes_bank_evaluation_metrics.png   # Model comparison chart
└── README.md
```

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```
Open `Yes_Bank_Stock_Prediction.ipynb` in Jupyter or Google Colab and run all cells.

---

## 🔗 Links

- **GitHub:** [Yes Bank Price Prediction](https://github.com/aankitsharma/Yes-Bank-Price---Supervised-ML-Model-)
- **Colab:** Open in Colab

---

## 👤 Author

**Aankit Sharma**
Advanced Certification in Full Stack Data Science & AI — AlmaBetter
12+ years experience in Financial Services | Transitioning to Data Science

[LinkedIn](#) | [GitHub](https://github.com/aankitsharma)
