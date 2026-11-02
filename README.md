# Predicting Bike-Sharing Demand

Predictive Analytics (CSE3141) project 2026, Manipal University Jaipur
Author: Atreya Mitra (Reg. No. 2427030002)

Classification, count regression and time-series forecasting on the Capital Bikeshare data (Washington D.C., 2011-2012).

## What is covered
- Data preparation and EDA: cleaning, impossible values, outliers, leakage check, VIF, mutual information, feature engineering
- Logistic regression and inference: binary logit, LDA, QDA, multinomial logit, Gaussian / Poisson / Negative Binomial GLMs
- Time series: decomposition, ADF test, ACF/PACF, ARIMA, SARIMA, exponential smoothing, SARIMAX
- Validation and optimisation: train-test split, stratified and time-ordered cross-validation, RMSE, MAE, F1, AUC-ROC, grid search, random search, over/under-fitting analysis

## Key results (held-out test data)
- Best classifier (gradient boosting, random-search tuned): AUC 0.986, F1 0.945
- Hourly counts are overdispersed (Poisson dispersion 14.3), so Negative Binomial is used for inference
- 60-day daily forecast: SARIMAX with weather and calendar regressors has the lowest RMSE (1,795)

## Dataset
UCI Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Run
```
pip install -r requirements.txt
python predictive_analytics_bike_demand.py
```
Put `hour.csv` and `day.csv` in a `data/` folder next to the script (it tries to download them if missing). Figures go to `figures/` and all numbers to `results.json`.

## Files
- `predictive_analytics_bike_demand.py`: full analysis code
- `Predictive_Analytics_Project_Report.docx`: project report
- `Predictive_Analytics_Presentation.pptx`: slides
