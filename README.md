# 🏡 Aura Estates: Advanced House Price Prediction

A production-ready machine learning pipeline and premium web application for real estate valuation, refactored from a beginner's notebook into a modular, scalable architecture.

![Luxury House Banner](https://i.imgur.com/your-generated-image-url.png) <!-- Note: Replace with actual hosted URL if possible, or leave as placeholder for user -->

## ✨ Key Features
- **Modular ML Pipeline**: Seperate modules for data loading, preprocessing, and modeling.
- **Top-Tier Performance**: Optimized XGBoost Regressor with hyperparameter tuning via Optuna.
- **Premium Web Interface**: A high-end FastAPI web app utilizing `Playfair Display` and `Source Sans Pro` typography.
- **Modern UI/UX**: Hover effects, responsive layout, and real-time inference.
- **Package Management**: Powered by `uv` for lightning-fast dependency resolution.

## 🏗️ Project Architecture
```text
.
├── data/raw/              # Original dataset CSVs (gitignored)
├── models/                # Saved model artifacts (joblib)
├── notebooks/             # Exploratory Data Analysis & original notebook
├── src/
│   ├── house_price_prediction/
│   │   ├── data_loader.py   # Dataset loading utilities
│   │   ├── preprocessing.py # Cleaning, Scaling, Encoding (MinMaxScaler, OneHot)
│   │   └── models.py        # Model definition, training & evaluation
│   ├── webapp/
│   │   ├── static/          # CSS, Images, JS
│   │   ├── templates/       # Jinja2 HTML templates
│   │   └── main.py          # FastAPI application
│   └── main.py              # CLI entry point for training
├── pyproject.toml           # Project dependencies & scripts
└── README.md
```

## 🚀 Getting Started

### 1. Installation
Ensure you have [uv](https://github.com/astral-sh/uv) installed.
```bash
uv sync
```

### 2. Training the Model
Run the main script to process data, evaluate base models, and train the optimized XGBoost.
```bash
uv run predict-prices
```

### 3. Launching the Web App
Start the FastAPI server to use the valuation engine.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run python src/webapp/main.py
```
Visit `http://localhost:8000` in your browser.

## 📊 Model Performance
The current best model (**XGBoost**) achieves an **RMSE of ~26,000**, significantly outperforming baseline linear and random forest models.

## 🛠️ Built With
- **FastAPI**: Backend web framework
- **Scikit-Learn**: Preprocessing & metrics
- **XGBoost**: Gradient boosted trees
- **Optuna**: Bayesian optimization
- **Pandas/Numpy**: Data manipulation
- **Jinja2**: Templating engine
- **UV**: Package manager

---
&copy; 2026 Aura Estates Prediction Engine. Created for MAS-Spring-2026.
