# NBA Player Points Predictor

A machine learning pipeline that predicts NBA player points per game using historical statistics. Built with Python, scikit-learn, and the `nba_api` library.

---

## Project Overview

This project demonstrates an end-to-end ML workflow:

1. **Data Collection** — Pull real NBA player stats via the `nba_api` library
2. **Feature Engineering** — Create rolling averages, usage estimates, and rest-day features
3. **Modeling** — Train and evaluate a Random Forest regression model
4. **Evaluation** — Report MAE, RMSE, and R² with a feature importance visualization

**Target variable:** Points scored in a game (`PTS`)

---

## Project Structure

```
nba-performance-predictor/
├── README.md
├── requirements.txt
├── data/
│   └── (auto-generated CSVs saved here)
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── data_loader.py         # Fetch and cache NBA stats
│   ├── features.py            # Feature engineering logic
│   └── model.py               # Model training and evaluation
└── main.py                    # Entry point — run the full pipeline
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/adamkarib/nba-performance-predictor.git
cd nba-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

This will:
- Fetch game logs for a set of NBA players
- Engineer features
- Train a Random Forest model
- Print evaluation metrics and save a feature importance plot

---

## Features Used

| Feature | Description |
|---|---|
| `rolling_pts_5` | Avg points over last 5 games |
| `rolling_reb_5` | Avg rebounds over last 5 games |
| `rolling_ast_5` | Avg assists over last 5 games |
| `rolling_min_5` | Avg minutes over last 5 games |
| `rolling_fga_5` | Avg field goal attempts over last 5 games |
| `rolling_fg_pct_5` | Avg FG% over last 5 games |
| `days_rest` | Days since last game |
| `home_game` | 1 if home game, 0 if away |

---
## Results

MAE:   6.89 points
RMSE:  8.83 points
R²:    -0.004

### Interpretation

The initial model showed **no predictive power** — an R² near zero means
it performs about as well as simply guessing each player's average points.

This is an expected result: **single-game scoring is dominated by noise**
(hot/cold shooting, blowouts, foul trouble), so the features available before
tip-off carry limited signal.

### Key takeaways
- The gap between RMSE (8.83) and MAE (6.89) indicates occasional large misses,
  not just small consistent errors.
- The strongest driver of points — minutes played — is only available as a
  lagging feature, limiting predictive power.

### Next steps
- Add projected minutes and opponent defensive rating as features
- Predict a more stable target (points per 36 minutes, or season average)
- Compare against XGBoost / LightGBM

---

## Tech Stack

- **Python 3.10+**
- **nba_api** — Free NBA stats API wrapper
- **pandas / numpy** — Data manipulation
- **scikit-learn** — Machine learning
- **matplotlib / seaborn** — Visualization

---

## Future Improvements

- Add opponent defensive rating as a feature
- Experiment with XGBoost or LightGBM
- Expand to predict assists, rebounds, or fantasy points
- Build a Streamlit dashboard for interactive predictions

---

## Author

**Adam Karibian** — MBAn | University of Michigan Ross School of Business  
Statistics + Mathematics B.S. | University of Michigan LSA
[LinkedIn] (https://www.linkedin.com/in/adam-karibian/)
