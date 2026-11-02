#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Analytics Project 2026
=================================
Title : Predicting Bike-Sharing Demand with Classification, Count Regression
        and Time-Series Forecasting (Capital Bikeshare, Washington D.C., 2011-2012)
Course: Predictive Analytics (CSE3141), Manipal University Jaipur

Everything in the syllabus is covered by one dataset:
  1. Introduction              -> problem framing (see report)
  2. Data preparation / EDA    -> section 1, 2
  3. Logistic regression & inference (LDA, QDA, multinomial logit, GLM,
     Poisson, Negative Binomial)                   -> section 3, 4
  4. Time-series analysis (ARIMA, SARIMA, exponential smoothing) -> section 5
  5. Model validation & optimisation (CV, RMSE, MAE, F1, AUC-ROC,
     grid / random search, over- and under-fitting) -> section 6

How to run
----------
    pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels
    python predictive_analytics_bike_demand.py

The script looks for data/hour.csv and data/day.csv next to itself. If they are
missing it downloads the UCI "Bike Sharing Dataset" (id 275) automatically.
Figures are written to ./figures and all numeric results to ./results.json.
"""
import io
import itertools
import json
import os
import urllib.request
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import loguniform, randint
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NAVY, ORANGE, GREY, TEAL, RED = "#1f4e79", "#e07a1f", "#8a8f98", "#2a9d8f", "#c0392b"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "axes.titleweight": "bold", "axes.titlesize": 11})

R = {}  # every number that appears in the report is stored here


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close()


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 0. Load data
# ----------------------------------------------------------------------------
def load_data():
    hp, dp = os.path.join(DATA, "hour.csv"), os.path.join(DATA, "day.csv")
    if not (os.path.exists(hp) and os.path.exists(dp)):
        urls = ["https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"]
        for u in urls:
            try:
                raw = urllib.request.urlopen(u, timeout=60).read()
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for n in ("hour.csv", "day.csv"):
                        open(os.path.join(DATA, n), "wb").write(z.read(n))
                break
            except Exception as e:  # noqa
                print("download failed:", e)
    return (pd.read_csv(hp, parse_dates=["dteday"]),
            pd.read_csv(dp, parse_dates=["dteday"]))


section("0. LOAD DATA")
hour, day = load_data()
R["n_hour"], R["n_day"] = len(hour), len(day)
R["date_min"], R["date_max"] = str(hour.dteday.min().date()), str(hour.dteday.max().date())
print(hour.shape, day.shape, R["date_min"], R["date_max"])

# ----------------------------------------------------------------------------
# 1. Data preparation
# ----------------------------------------------------------------------------
section("1. DATA PREPARATION")
R["missing_total"] = int(hour.isna().sum().sum())
R["duplicates"] = int(hour.duplicated().sum())
R["dtypes"] = {c: str(t) for c, t in hour.dtypes.items()}
R["describe_cnt"] = hour["cnt"].describe().round(2).to_dict()
# the hourly file has no gaps? (24 h x 731 days = 17544 but 17379 rows exist)
R["expected_hours"] = int(731 * 24)
R["missing_hours"] = R["expected_hours"] - len(hour)

# The UCI file stores weather variables normalised -> convert back to real units
for df in (hour, day):
    df["temp_c"] = df["temp"] * 41          # degrees Celsius
    df["atemp_c"] = df["atemp"] * 50        # feels-like, Celsius
    df["hum_pct"] = df["hum"] * 100         # % relative humidity
    df["wind_kmh"] = df["windspeed"] * 67   # km/h

# (a) Implausible values: humidity = 0 is physically impossible in D.C.
R["hum_zero_hour"] = int((hour.hum_pct == 0).sum())
R["hum_zero_day"] = int((day.hum_pct == 0).sum())
R["hum_zero_dates"] = sorted({str(d.date()) for d in hour.loc[hour.hum_pct == 0, "dteday"]})
for df in (hour, day):
    df.loc[df.hum_pct == 0, "hum_pct"] = np.nan
    df["hum_pct"] = df.groupby(["yr", "mnth"])["hum_pct"].transform(lambda s: s.fillna(s.median()))
R["hum_missing_after"] = int(hour.hum_pct.isna().sum())

# (b) Outliers: IQR rule
def iqr_fence(s):
    q1, q3 = s.quantile([.25, .75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

out = {}
for c in ["cnt", "wind_kmh", "hum_pct", "temp_c"]:
    lo, hi = iqr_fence(hour[c])
    out[c] = {"low": round(lo, 2), "high": round(hi, 2),
              "n_out": int(((hour[c] < lo) | (hour[c] > hi)).sum())}
R["outliers"] = out
R["wind_zero"] = int((hour.wind_kmh == 0).sum())
wind_before = hour.wind_kmh.copy()
_, wind_hi = iqr_fence(hour.wind_kmh)
hour["wind_kmh"] = hour.wind_kmh.clip(upper=wind_hi)   # winsorise genuine extreme gusts
day["wind_kmh"] = day.wind_kmh.clip(upper=iqr_fence(day.wind_kmh)[1])
# demand outliers are real rush-hour peaks -> kept (removing them would remove the signal)

# (c) Feature engineering
for df in (hour, day):
    df["weather_cat"] = df["weathersit"].clip(upper=3)        # merge 3 heavy-rain rows with "bad"
    df["weather_mist"] = (df.weather_cat == 2).astype(int)
    df["weather_bad"] = (df.weather_cat == 3).astype(int)
hour["hr_sin"] = np.sin(2 * np.pi * hour.hr / 24)
hour["hr_cos"] = np.cos(2 * np.pi * hour.hr / 24)
hour["hr_sin2"] = np.sin(4 * np.pi * hour.hr / 24)         # 12-hour harmonic -> two commuting peaks
hour["hr_cos2"] = np.cos(4 * np.pi * hour.hr / 24)
for h in ("hr_sin", "hr_cos", "hr_sin2", "hr_cos2"):
    hour["wd_" + h] = hour.workingday * hour[h]             # different daily shape on working days
hour["hr_daytype"] = hour.hr.astype(str) + "_" + hour.workingday.astype(str)

# Binary target (median split) and 3-level target (tertiles)
THRESH = float(hour.cnt.median())
hour["high"] = (hour.cnt > THRESH).astype(int)
hour["level"] = pd.qcut(hour.cnt, 3, labels=[0, 1, 2]).astype(int)
R["threshold_median"] = THRESH
R["tertile_cuts"] = [float(x) for x in hour.cnt.quantile([1 / 3, 2 / 3])]
R["class_balance"] = hour.high.value_counts(normalize=True).round(3).to_dict()

# (d) Leakage check: cnt = casual + registered, so those two columns are NOT features
R["leak_check"] = bool((hour.casual + hour.registered == hour.cnt).all())

FEATS = ["temp_c", "hum_pct", "wind_kmh", "workingday", "holiday", "yr",
         "weather_mist", "weather_bad", "hr_sin", "hr_cos", "hr_sin2", "hr_cos2",
         "wd_hr_sin", "wd_hr_cos", "wd_hr_sin2", "wd_hr_cos2"]
FEATS_TREE = ["temp_c", "hum_pct", "wind_kmh", "workingday", "holiday", "yr",
              "weather_cat", "hr", "mnth", "weekday", "season"]

# (e) Feature selection: multicollinearity (VIF) and mutual information
vif_cols = ["temp_c", "atemp_c", "hum_pct", "wind_kmh", "workingday", "yr"]
Xv = sm.add_constant(hour[vif_cols])
R["vif_before"] = {c: round(float(variance_inflation_factor(Xv.values, i)), 2)
                   for i, c in enumerate(Xv.columns) if c != "const"}
Xv2 = sm.add_constant(hour[[c for c in vif_cols if c != "atemp_c"]])
R["vif_after"] = {c: round(float(variance_inflation_factor(Xv2.values, i)), 2)
                  for i, c in enumerate(Xv2.columns) if c != "const"}
R["corr_temp_atemp"] = round(float(hour.temp_c.corr(hour.atemp_c)), 4)
mi = mutual_info_classif(hour[FEATS_TREE], hour.high, discrete_features=[3, 4, 5, 6, 7, 8, 9, 10], random_state=SEED)
R["mutual_info"] = dict(sorted({c: round(float(v), 4) for c, v in zip(FEATS_TREE, mi)}.items(),
                               key=lambda kv: -kv[1]))
print("VIF before:", R["vif_before"], "\nVIF after :", R["vif_after"])
print("MI:", R["mutual_info"])

# One common split used by every supervised task below
idx_tr, idx_te = train_test_split(hour.index, test_size=0.2, random_state=SEED, stratify=hour.high)
train, test = hour.loc[idx_tr].copy(), hour.loc[idx_te].copy()
R["n_train"], R["n_test"] = len(train), len(test)

# ----------------------------------------------------------------------------
# 2. Exploratory data analysis
# ----------------------------------------------------------------------------
section("2. EDA")
# Fig 1: distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(hour.cnt, bins=50, color=NAVY, edgecolor="white")
ax[0].axvline(THRESH, color=ORANGE, ls="--", lw=2, label=f"median = {THRESH:.0f}")
ax[0].set(title="Hourly rentals are right-skewed", xlabel="Rentals per hour (cnt)", ylabel="Hours")
ax[0].legend()
ax[1].scatter(hour.cnt.groupby(hour.hr).mean(), hour.cnt.groupby(hour.hr).var(), color=NAVY)
m = np.linspace(20, 450, 50)
ax[1].plot(m, m, color=ORANGE, ls="--", label="Poisson: variance = mean")
ax[1].set(title="Variance far exceeds the mean (overdispersion)", xlabel="Mean rentals by hour of day",
          ylabel="Variance")
ax[1].legend()
savefig("fig01_distribution.png")

# Fig 2: hourly profile
prof = hour.groupby(["hr", "workingday"]).cnt.mean().unstack()
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(prof.index, prof[1], color=NAVY, lw=2.5, marker="o", ms=4, label="Working day")
ax.plot(prof.index, prof[0], color=ORANGE, lw=2.5, marker="o", ms=4, label="Weekend / holiday")
ax.set(title="Daily demand shape: commuter peaks vs leisure hump", xlabel="Hour of day",
       ylabel="Average rentals", xticks=range(0, 24, 2))
ax.legend()
savefig("fig02_hourly_profile.png")
R["peak_hours_work"] = [int(h) for h in prof[1].nlargest(2).index]
R["peak_work_vals"] = [round(float(v), 1) for v in prof[1].nlargest(2)]
R["peak_hour_nonwork"] = int(prof[0].idxmax())
R["peak_nonwork_val"] = round(float(prof[0].max()), 1)

# Fig 3: monthly trend 2011 vs 2012
mm = day.groupby(["yr", "mnth"]).cnt.mean().unstack(0)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(mm.index, mm[0], color=GREY, lw=2.5, marker="o", label="2011")
ax.plot(mm.index, mm[1], color=NAVY, lw=2.5, marker="o", label="2012")
ax.set(title="Seasonality and growth: average daily rentals by month", xlabel="Month",
       ylabel="Average daily rentals", xticks=range(1, 13))
ax.legend()
savefig("fig03_monthly.png")
R["growth_2012_vs_2011_pct"] = round(float((day[day.yr == 1].cnt.mean() / day[day.yr == 0].cnt.mean() - 1) * 100), 1)
R["mean_daily_2011"] = round(float(day[day.yr == 0].cnt.mean()), 0)
R["mean_daily_2012"] = round(float(day[day.yr == 1].cnt.mean()), 0)

# Fig 4: weather and temperature
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=hour, x="weather_cat", y="cnt", ax=ax[0], palette=[NAVY, TEAL, ORANGE], fliersize=2)
ax[0].set(title="Worse weather, lower demand", xlabel="Weather (1 clear, 2 mist, 3 rain/snow)", ylabel="Rentals per hour")
ax[1].scatter(hour.temp_c, hour.cnt, s=4, alpha=.25, color=NAVY)
z = np.polyfit(hour.temp_c, hour.cnt, 2)
xs = np.linspace(hour.temp_c.min(), hour.temp_c.max(), 100)
ax[1].plot(xs, np.polyval(z, xs), color=ORANGE, lw=2.5)
ax[1].set(title="Demand rises with temperature", xlabel="Temperature (°C)", ylabel="Rentals per hour")
savefig("fig04_weather.png")
R["weather_means"] = hour.groupby("weather_cat").cnt.mean().round(1).to_dict()
R["weather_counts"] = hour.weather_cat.value_counts().sort_index().to_dict()

# Fig 5: correlation heatmap
cc = ["cnt", "temp_c", "atemp_c", "hum_pct", "wind_kmh", "hr", "workingday", "yr", "casual", "registered"]
corr = hour[cc].corr()
fig, ax = plt.subplots(figsize=(7, 5.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})
ax.set_title("Correlation matrix")
savefig("fig05_correlation.png")
R["corr_cnt"] = corr["cnt"].round(3).to_dict()

# Fig 6: outlier handling
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
ax[0].boxplot(hour.cnt, vert=True, flierprops=dict(markersize=2, markerfacecolor=NAVY))
ax[0].set(title="cnt: peaks are real (kept)", xticks=[], ylabel="Rentals per hour")
ax[1].boxplot([wind_before, hour.wind_kmh], flierprops=dict(markersize=2, markerfacecolor=NAVY))
ax[1].set_xticklabels(["raw", "winsorised"])
ax[1].set(title="Wind speed: extreme gusts capped", ylabel="km/h")
hh = hour[hour.dteday == "2011-03-10"]
ax[2].plot(hh.hr, [0] * len(hh), color=RED, marker="o", label="recorded (0 %)")
ax[2].plot(hh.hr, hh.hum_pct, color=NAVY, marker="o", label="imputed")
ax[2].set(title="Humidity = 0 on 2011-03-10 fixed", xlabel="Hour", ylabel="% humidity")
ax[2].legend()
savefig("fig06_outliers.png")

# ----------------------------------------------------------------------------
# 3. Logistic regression, discriminant analysis, multinomial logit
# ----------------------------------------------------------------------------
section("3. CLASSIFICATION")
f_logit = "high ~ " + " + ".join(FEATS)
logit = smf.logit(f_logit, train).fit(disp=0, maxiter=200)
ci = logit.conf_int()
lt = pd.DataFrame({"coef": logit.params, "se": logit.bse, "z": logit.tvalues, "p": logit.pvalues,
                   "OR": np.exp(logit.params), "OR_lo": np.exp(ci[0]), "OR_hi": np.exp(ci[1])})
R["logit_table"] = lt.round(4).reset_index().rename(columns={"index": "term"}).to_dict("records")
R["logit_llf"] = round(float(logit.llf), 1)
R["logit_llnull"] = round(float(logit.llnull), 1)
R["logit_pseudo_r2"] = round(float(logit.prsquared), 4)
R["logit_lr_stat"] = round(float(logit.llr), 1)
R["logit_lr_p"] = float(logit.llr_pvalue)
R["logit_aic"] = round(float(logit.aic), 1)
print(lt.round(3))

# scaled odds ratios for interpretable variables
scale = {"temp_c": 5, "hum_pct": 10, "wind_kmh": 10}
orows = []
for t in ["temp_c", "hum_pct", "wind_kmh", "holiday", "yr", "weather_mist", "weather_bad"]:
    k = scale.get(t, 1)
    orows.append({"term": t, "per": k, "OR": float(np.exp(k * logit.params[t])),
                  "lo": float(np.exp(k * ci.loc[t, 0])), "hi": float(np.exp(k * ci.loc[t, 1])),
                  "p": float(logit.pvalues[t])})
R["odds_scaled"] = orows
labels = {"temp_c": "+5 °C", "hum_pct": "+10 % humidity", "wind_kmh": "+10 km/h wind",
          "holiday": "Holiday", "yr": "Year 2012 vs 2011", "weather_mist": "Mist vs clear",
          "weather_bad": "Rain/snow vs clear"}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, r in enumerate(orows[::-1]):
    ax.plot([r["lo"], r["hi"]], [i, i], color=NAVY, lw=2.5)
    ax.scatter(r["OR"], i, color=ORANGE, s=45, zorder=3)
ax.axvline(1, color=GREY, ls="--")
ax.set_yticks(range(len(orows)))
ax.set_yticklabels([labels[r["term"]] for r in orows[::-1]])
ax.set_xscale("log")
ax.set(title="Odds of a 'high-demand' hour (95 % CI)", xlabel="Odds ratio (log scale)")
savefig("fig07_odds_ratios.png")

# --- sklearn pipelines on the same split -----------------------------------------------
Xtr, ytr, Xte, yte = train[FEATS], train.high, test[FEATS], test.high
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)

def scaled(m):
    return Pipeline([("sc", StandardScaler()), ("m", m)])

models = {
    "Logistic regression": scaled(LogisticRegression(max_iter=3000)),
    "LDA": scaled(LinearDiscriminantAnalysis()),
    "QDA": scaled(QuadraticDiscriminantAnalysis(reg_param=0.1)),
}
clf_rows, probs = [], {}
for name, mdl in models.items():
    cvf1 = cross_val_score(mdl, Xtr, ytr, cv=cv5, scoring="f1")
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    yh = (p >= .5).astype(int)
    probs[name] = p
    clf_rows.append({"model": name, "cv_f1": cvf1.mean(), "cv_f1_sd": cvf1.std(),
                     "acc": accuracy_score(yte, yh), "prec": precision_score(yte, yh),
                     "rec": recall_score(yte, yh), "f1": f1_score(yte, yh), "auc": roc_auc_score(yte, p)})
    print(name, {k: round(v, 4) for k, v in clf_rows[-1].items() if k != "model"})

# Random forest (default) as non-linear reference
rf0 = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
Xtr_t, Xte_t = train[FEATS_TREE], test[FEATS_TREE]
cvf1 = cross_val_score(rf0, Xtr_t, ytr, cv=cv5, scoring="f1")
rf0.fit(Xtr_t, ytr)
p = rf0.predict_proba(Xte_t)[:, 1]
yh = (p >= .5).astype(int)
probs["Random forest (default)"] = p
clf_rows.append({"model": "Random forest (default)", "cv_f1": cvf1.mean(), "cv_f1_sd": cvf1.std(),
                 "acc": accuracy_score(yte, yh), "prec": precision_score(yte, yh),
                 "rec": recall_score(yte, yh), "f1": f1_score(yte, yh), "auc": roc_auc_score(yte, p)})
R["lda_priors"] = [round(float(x), 3) for x in models["LDA"].named_steps["m"].priors_]
R["qda_priors"] = [round(float(x), 3) for x in models["QDA"].named_steps["m"].priors_]
lda_w = pd.Series(models["LDA"].named_steps["m"].coef_[0], index=FEATS)
R["lda_top_coefs"] = lda_w.reindex(lda_w.abs().sort_values(ascending=False).index).head(6).round(3).to_dict()

# --- multinomial logit (3 demand levels) ---------------------------------------------------
Xtr_c = sm.add_constant(train[FEATS].astype(float))
Xte_c = sm.add_constant(test[FEATS].astype(float), has_constant="add")
try:
    mn = sm.MNLogit(train.level, Xtr_c).fit(disp=0, maxiter=300, method="newton")
except Exception:
    mn = sm.MNLogit(train.level, Xtr_c).fit(disp=0, maxiter=2000, method="bfgs")
pm = mn.predict(Xte_c)
y3 = pm.values.argmax(1)
R["mn_acc"] = float(accuracy_score(test.level, y3))
R["mn_macro_f1"] = float(f1_score(test.level, y3, average="macro"))
R["mn_llf"], R["mn_pseudo_r2"] = round(float(mn.llf), 1), round(float(mn.prsquared), 4)
cm3 = pd.crosstab(test.level, y3).values
R["mn_confusion"] = cm3.tolist()
mnp = np.exp(mn.params)
R["mn_or"] = {str(c): {k: round(float(mnp.loc[k, c]), 4) for k in
                       ["temp_c", "hum_pct", "wind_kmh", "holiday", "yr", "weather_mist", "weather_bad"]}
              for c in mnp.columns}
R["mn_p"] = {str(c): {k: float(mn.pvalues.loc[k, c]) for k in
                      ["temp_c", "hum_pct", "wind_kmh", "holiday", "yr", "weather_mist", "weather_bad"]}
             for c in mn.pvalues.columns}
print("Multinomial acc", R["mn_acc"], "macro F1", R["mn_macro_f1"])

# ----------------------------------------------------------------------------
# 4. Generalised linear models for counts: Gaussian vs Poisson vs Negative Binomial
# ----------------------------------------------------------------------------
section("4. COUNT REGRESSION / GLM")
f_glm = ("cnt ~ C(hr_daytype) + temp_c + hum_pct + wind_kmh + holiday + yr + "
         "C(weather_cat) + C(season)")
gau = smf.glm(f_glm, train, family=sm.families.Gaussian()).fit()
poi = smf.glm(f_glm, train, family=sm.families.Poisson()).fit()
R["poisson_dispersion"] = round(float(poi.pearson_chi2 / poi.df_resid), 2)
# Cameron-Trivedi auxiliary regression for the NB2 dispersion parameter alpha
mu = poi.mu
aux = ((train.cnt - mu) ** 2 - train.cnt) / mu
alpha = float(sm.OLS(aux, mu).fit().params.iloc[0])
nb = smf.glm(f_glm, train, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
R["nb_alpha"] = round(alpha, 4)
R["nb_dispersion"] = round(float(nb.pearson_chi2 / nb.df_resid), 2)

rf_reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
rf_reg.fit(train[FEATS_TREE], train.cnt)
glm_rows = []
for name, res in [("Gaussian GLM (OLS)", gau), ("Poisson GLM", poi), ("Negative Binomial GLM", nb)]:
    pt, pe = np.clip(res.predict(train), 0, None), np.clip(res.predict(test), 0, None)
    glm_rows.append({"model": name, "aic": float(res.aic), "dev": float(res.deviance),
                     "rmse_train": rmse(train.cnt, pt), "rmse": rmse(test.cnt, pe),
                     "mae": float(mean_absolute_error(test.cnt, pe)),
                     "disp": float(res.pearson_chi2 / res.df_resid)})
pe = rf_reg.predict(test[FEATS_TREE])
glm_rows.append({"model": "Random forest regressor", "aic": None, "dev": None,
                 "rmse_train": rmse(train.cnt, rf_reg.predict(train[FEATS_TREE])),
                 "rmse": rmse(test.cnt, pe), "mae": float(mean_absolute_error(test.cnt, pe)), "disp": None})
R["glm_rows"] = glm_rows
for r in glm_rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

# rate ratios from the NB model
ci_nb = nb.conf_int()
rr = []
for t, k, lab in [("temp_c", 1, "Temperature, +1 °C"), ("hum_pct", 10, "Humidity, +10 %"),
                  ("wind_kmh", 10, "Wind speed, +10 km/h"), ("holiday", 1, "Holiday"),
                  ("yr", 1, "Year 2012 vs 2011"),
                  ("C(weather_cat)[T.2]", 1, "Mist vs clear"), ("C(weather_cat)[T.3]", 1, "Rain/snow vs clear"),
                  ("C(season)[T.2]", 1, "Spring vs winter (season 2 vs 1)"),
                  ("C(season)[T.3]", 1, "Summer vs winter (season 3 vs 1)"),
                  ("C(season)[T.4]", 1, "Fall vs winter (season 4 vs 1)")]:
    rr.append({"term": lab, "RR": float(np.exp(k * nb.params[t])), "lo": float(np.exp(k * ci_nb.loc[t, 0])),
               "hi": float(np.exp(k * ci_nb.loc[t, 1])), "p": float(nb.pvalues[t])})
R["nb_rr"] = rr
# same effects under Poisson to show the standard-error problem
R["se_ratio_nb_over_poisson"] = round(float((nb.bse["temp_c"]) / (poi.bse["temp_c"])), 2)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
pn = np.clip(nb.predict(test), 0, None)
ax[0].scatter(test.cnt, pn, s=4, alpha=.3, color=NAVY)
ax[0].plot([0, 900], [0, 900], color=ORANGE, ls="--")
ax[0].set(title="Negative Binomial GLM: predicted vs actual (test)", xlabel="Actual rentals", ylabel="Predicted")
names = [r["model"].replace(" GLM", "").replace(" regressor", "") for r in glm_rows]
ax[1].bar(names, [r["rmse"] for r in glm_rows], color=[GREY, GREY, NAVY, TEAL])
for i, r in enumerate(glm_rows):
    ax[1].text(i, r["rmse"] + 1, f"{r['rmse']:.1f}", ha="center", fontsize=9)
ax[1].set(title="Test RMSE by model family", ylabel="RMSE (rentals/hour)")
plt.setp(ax[1].get_xticklabels(), rotation=15)
savefig("fig08_glm.png")

# ----------------------------------------------------------------------------
# 6 (first half). Validation and hyper-parameter optimisation for classification
# ----------------------------------------------------------------------------
section("6. VALIDATION AND TUNING")
# Grid search: random forest
grid = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1),
                    {"n_estimators": [100, 200], "max_depth": [6, 12, 20, None],
                     "min_samples_leaf": [1, 5, 10]},
                    cv=cv5, scoring="f1", n_jobs=2)
grid.fit(Xtr_t, ytr)
R["grid_best_params"] = {k: (None if v is None else int(v)) for k, v in grid.best_params_.items()}
R["grid_best_cv_f1"] = float(grid.best_score_)
R["grid_n_combos"] = int(len(grid.cv_results_["params"]))
R["grid_default_cv_f1"] = float(clf_rows[-1]["cv_f1"])
rf_best = grid.best_estimator_

# Random search: histogram-based gradient boosting
rs = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=SEED),
                        {"learning_rate": loguniform(0.02, 0.3), "max_depth": randint(3, 10),
                         "max_leaf_nodes": randint(8, 64), "min_samples_leaf": randint(10, 60),
                         "l2_regularization": loguniform(1e-3, 10)},
                        n_iter=25, cv=cv5, scoring="f1", random_state=SEED, n_jobs=2)
rs.fit(Xtr_t, ytr)
R["rand_best_params"] = {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in rs.best_params_.items()}
R["rand_best_cv_f1"] = float(rs.best_score_)
gb_best = rs.best_estimator_
print("grid:", grid.best_params_, grid.best_score_, "| random:", rs.best_params_, rs.best_score_)

for name, mdl in [("Random forest (grid-tuned)", rf_best), ("Gradient boosting (random-tuned)", gb_best)]:
    p = mdl.predict_proba(Xte_t)[:, 1]
    yh = (p >= .5).astype(int)
    probs[name] = p
    clf_rows.append({"model": name, "cv_f1": float(grid.best_score_ if "forest" in name else rs.best_score_),
                     "cv_f1_sd": None, "acc": accuracy_score(yte, yh), "prec": precision_score(yte, yh),
                     "rec": recall_score(yte, yh), "f1": f1_score(yte, yh), "auc": roc_auc_score(yte, p)})
R["clf_rows"] = clf_rows
for r in clf_rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

best_name = max(clf_rows, key=lambda r: r["auc"])["model"]
R["best_model"] = best_name

# ROC curves
fig, ax = plt.subplots(figsize=(5.8, 5))
cols = [GREY, TEAL, "#7b6fb0", "#b8b8b8", NAVY, ORANGE]
for (n, p), c in zip(probs.items(), cols):
    fpr, tpr, _ = roc_curve(yte, p)
    ax.plot(fpr, tpr, lw=2.2 if n == best_name else 1.5, color=c, label=f"{n} (AUC {roc_auc_score(yte, p):.3f})")
ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1)
ax.set(title="ROC curves on the held-out test set", xlabel="False positive rate", ylabel="True positive rate")
ax.legend(fontsize=7.5, loc="lower right")
savefig("fig09_roc.png")

# Confusion matrix for the best model
pb = probs[best_name]
yb = (pb >= .5).astype(int)
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ConfusionMatrixDisplay.from_predictions(yte, yb, display_labels=["Low", "High"], cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix: {best_name}", fontsize=9)
ax.grid(False)
savefig("fig10_confusion.png")
tn, fp, fn, tp = [int(v) for v in np.ravel(__import__("sklearn.metrics", fromlist=["x"]).confusion_matrix(yte, yb))]
R["best_confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

# Feature importance (tuned RF)
imp = pd.Series(rf_best.feature_importances_, index=FEATS_TREE).sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set(title="What drives high demand? (tuned random forest)", xlabel="Importance (mean decrease in impurity)")
savefig("fig11_importance.png")
R["rf_importance"] = imp.sort_values(ascending=False).round(4).to_dict()

# --- CV schemes: random K-fold vs time-ordered folds ---------------------------------------
Xall, yall = hour[FEATS_TREE], hour.high
rf_cv = RandomForestClassifier(random_state=SEED, n_jobs=2, **{k: v for k, v in grid.best_params_.items()})
r_cv = cross_val_score(rf_cv, Xall, yall, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1")
t_cv = cross_val_score(rf_cv, Xall, yall, cv=TimeSeriesSplit(5), scoring="f1")
R["cv_random_f1"] = [float(x) for x in r_cv]
R["cv_time_f1"] = [float(x) for x in t_cv]
# Chronological hold-out: train on first 75 % of the calendar, test on the rest
cut = int(len(hour) * .75)
rf_cv.fit(hour.iloc[:cut][FEATS_TREE], hour.iloc[:cut].high)
pt = rf_cv.predict_proba(hour.iloc[cut:][FEATS_TREE])[:, 1]
R["temporal_holdout"] = {"cut_date": str(hour.iloc[cut].dteday.date()),
                         "f1": float(f1_score(hour.iloc[cut:].high, pt > .5)),
                         "auc": float(roc_auc_score(hour.iloc[cut:].high, pt))}
print("random CV F1", r_cv.mean(), "| time-series CV F1", t_cv.mean(), "| holdout", R["temporal_holdout"])

# --- Over- / under-fitting ------------------------------------------------------------------
depths = list(range(1, 26))
tr_s, va_s = validation_curve(DecisionTreeClassifier(random_state=SEED), Xtr_t, ytr, param_name="max_depth",
                              param_range=depths, cv=cv5, scoring="f1", n_jobs=2)
R["dt_curve"] = {"depth": depths, "train": tr_s.mean(1).round(4).tolist(), "cv": va_s.mean(1).round(4).tolist()}
best_d = depths[int(va_s.mean(1).argmax())]
R["dt_best_depth"] = best_d
sizes, ltr, lva = learning_curve(DecisionTreeClassifier(random_state=SEED), Xtr_t, ytr, cv=cv5, scoring="f1",
                                 train_sizes=np.linspace(.1, 1, 6), n_jobs=2)
sizes2, ltr2, lva2 = learning_curve(DecisionTreeClassifier(max_depth=best_d, random_state=SEED), Xtr_t, ytr,
                                    cv=cv5, scoring="f1", train_sizes=np.linspace(.1, 1, 6), n_jobs=2)
Cs = np.logspace(-4, 1, 11)
l1_cv, l1_nz = [], []
for C in Cs:
    mm_ = scaled(LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=3000))
    l1_cv.append(cross_val_score(mm_, Xtr, ytr, cv=cv5, scoring="f1").mean())
    l1_nz.append(int((mm_.fit(Xtr, ytr).named_steps["m"].coef_ != 0).sum()))
R["l1_path"] = {"C": [float(c) for c in Cs], "cv_f1": [float(x) for x in l1_cv], "nonzero": l1_nz}

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
ax[0].plot(depths, R["dt_curve"]["train"], color=ORANGE, lw=2.2, label="Training F1")
ax[0].plot(depths, R["dt_curve"]["cv"], color=NAVY, lw=2.2, label="Cross-validated F1")
ax[0].axvline(best_d, color=GREY, ls="--")
ax[0].set(title="Tree depth: shallow = underfit, deep = overfit", xlabel="max_depth", ylabel="F1")
ax[0].legend(loc="lower right")
ax[1].plot(sizes, ltr.mean(1), color=ORANGE, marker="o", label="Train (unpruned)")
ax[1].plot(sizes, lva.mean(1), color=NAVY, marker="o", label="CV (unpruned)")
ax[1].plot(sizes2, lva2.mean(1), color=TEAL, marker="s", ls="--", label=f"CV (depth {best_d})")
ax[1].set(title="Learning curve: gap = variance", xlabel="Training examples", ylabel="F1")
ax[1].legend(fontsize=8, loc="lower right")
ax[2].semilogx(Cs, l1_cv, color=NAVY, marker="o", lw=2)
ax[2].set(title="L1 regularisation (logistic)", xlabel="C (inverse penalty)", ylabel="CV F1")
a2 = ax[2].twinx()
a2.plot(Cs, l1_nz, color=ORANGE, marker="s", ls="--")
a2.set_ylabel("Non-zero coefficients", color=ORANGE)
a2.grid(False)
savefig("fig12_overfitting.png")
R["learning_gap_unpruned"] = round(float(ltr.mean(1)[-1] - lva.mean(1)[-1]), 4)
R["learning_gap_pruned"] = round(float(ltr2.mean(1)[-1] - lva2.mean(1)[-1]), 4)

# ----------------------------------------------------------------------------
# 5. Time-series analysis (daily demand)
# ----------------------------------------------------------------------------
section("5. TIME SERIES")
ts = day.set_index("dteday")["cnt"].astype(float).asfreq("D")
R["ts_missing_days"] = int(ts.isna().sum())
H = 60
OUT_Z = 6.0

# Outlier handling for the series: a robust rule (deviation from a 15-day centred rolling median,
# modified z-score > 6 on the MAD scale (only high-impact events)). Flagged days (e.g. Hurricane Sandy, 28-30 Oct 2012,
# and heavy snow days) are replaced by linear interpolation in the TRAINING data only;
# the hold-out period is always scored against the raw, real observations.
ts_raw = ts.copy()
dev = ts_raw - ts_raw.rolling(15, center=True, min_periods=8).median()
mad = np.median(np.abs(dev - dev.median()))
rz = 0.6745 * (dev - dev.median()) / mad
ts_out = rz.abs() > OUT_Z
ts_clean = ts_raw.mask(ts_out).interpolate(limit_direction="both")
R["ts_outliers"] = [{"date": str(d.date()), "value": float(ts_raw[d]), "z": round(float(rz[d]), 1)}
                    for d in ts_raw.index[ts_out]]
print("flagged days:", R["ts_outliers"])
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(ts_raw.index, ts_raw.values, color=NAVY, lw=1.2, label="Observed daily rentals")
ax.scatter(ts_raw.index[ts_out], ts_raw[ts_out], color=RED, zorder=3, s=28, label="Flagged outlier")
ax.set(title="Outlier detection on the daily series (Hurricane Sandy: 29 Oct 2012 = 22 rentals)",
       ylabel="Rentals per day")
ax.legend(loc="upper left", fontsize=8)
savefig("fig13b_ts_outliers.png")

ts = ts_clean                              # all modelling below uses the cleaned series
tr_ts, te_ts = ts_clean.iloc[:-H], ts_raw.iloc[-H:]
R["ts_train_range"] = [str(tr_ts.index[0].date()), str(tr_ts.index[-1].date())]
R["ts_test_range"] = [str(te_ts.index[0].date()), str(te_ts.index[-1].date())]

dec = seasonal_decompose(ts, model="additive", period=7)
fig, ax = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
for a, s, t in zip(ax, [ts, dec.trend, dec.seasonal, dec.resid], ["Observed", "Trend (7-day centred MA)", "Weekly seasonal component", "Residual (noise)"]):
    a.plot(s.index, s.values, color=NAVY, lw=1.2)
    a.set_ylabel(t, fontsize=8)
savefig("fig13_decomposition.png")
sea = dec.seasonal.iloc[:7]
R["weekly_seasonal"] = {str(d.day_name())[:3]: round(float(v), 1) for d, v in sea.items()}
R["decomp_var_share"] = {"trend": round(float(dec.trend.dropna().var() / ts.loc[dec.trend.dropna().index].var()), 3),
                         "seasonal": round(float(dec.seasonal.var() / ts.var()), 4),
                         "resid": round(float(dec.resid.dropna().var() / ts.loc[dec.resid.dropna().index].var()), 3)}

def adf(s):
    r = adfuller(s.dropna(), autolag="AIC")
    return {"stat": round(float(r[0]), 3), "p": round(float(r[1]), 4)}

R["adf"] = {"level": adf(tr_ts), "diff1": adf(tr_ts.diff(1)), "seasonal_diff7": adf(tr_ts.diff(7)),
            "diff1_and_diff7": adf(tr_ts.diff(1).diff(7))}
print("ADF", R["adf"])

fig, ax = plt.subplots(2, 2, figsize=(10, 5.6))
plot_acf(tr_ts, lags=40, ax=ax[0, 0], color=NAVY, title="ACF: original series")
plot_pacf(tr_ts, lags=40, ax=ax[0, 1], color=NAVY, title="PACF: original series", method="ywm")
d2 = tr_ts.diff(1).diff(7).dropna()
plot_acf(d2, lags=40, ax=ax[1, 0], color=NAVY, title="ACF: after d=1 and weekly D=1")
plot_pacf(d2, lags=40, ax=ax[1, 1], color=NAVY, title="PACF: after d=1 and weekly D=1", method="ywm")
savefig("fig14_acf_pacf.png")

ex = day.set_index("dteday")[["temp_c", "hum_pct", "wind_kmh", "holiday", "weather_bad", "workingday"]].astype(float).asfreq("D")
ex["xmas"] = ((ex.index.month == 12) & (ex.index.day >= 22)).astype(float)   # known calendar effect: 22-31 Dec
ex["trend"] = np.arange(len(ex)) / 365.0                                      # captures the 2011 -> 2012 growth
exog_cols = list(ex.columns)
ex_tr, ex_te = ex.iloc[:-H], ex.iloc[-H:]

fc, rows = {}, []
def record(name, pred, extra=None):
    pred = pd.Series(np.asarray(pred, float), index=te_ts.index)
    fc[name] = pred
    rows.append({"model": name, "rmse": rmse(te_ts, pred), "mae": float(mean_absolute_error(te_ts, pred)),
                 "mape": mape(te_ts, pred), **(extra or {})})

# baselines
record("Seasonal naive (t-7)", np.tile(tr_ts.iloc[-7:].values, int(np.ceil(H / 7)))[:H])
record("Naive (last value)", np.repeat(tr_ts.iloc[-1], H))

# ARIMA grid
arima_aic = {}
for p, q in itertools.product(range(4), range(4)):
    try:
        arima_aic[(p, 1, q)] = ARIMA(tr_ts, order=(p, 1, q)).fit().aic
    except Exception:
        pass
ao = min(arima_aic, key=arima_aic.get)
arima = ARIMA(tr_ts, order=ao).fit()
record(f"ARIMA{ao}", arima.forecast(H), {"aic": float(arima.aic)})
R["arima_order"], R["arima_grid_n"] = list(ao), len(arima_aic)

# SARIMA grid (weekly season)
sar_aic = {}
for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
    try:
        r_ = SARIMAX(tr_ts, order=(p, 1, q), seasonal_order=(P, 1, Q, 7)).fit(disp=False, maxiter=60)
        sar_aic[(p, 1, q, P, 1, Q)] = r_.aic
    except Exception:
        pass
so = min(sar_aic, key=sar_aic.get)
sarima = SARIMAX(tr_ts, order=so[:3], seasonal_order=(so[3], so[4], so[5], 7)).fit(disp=False, maxiter=200)
record(f"SARIMA{so[:3]}x{(so[3], so[4], so[5], 7)}", sarima.get_forecast(H).predicted_mean, {"aic": float(sarima.aic)})
R["sarima_order"], R["sarima_grid_n"] = list(so), len(sar_aic)
R["sarima_name"] = rows[-1]["model"]
lb = acorr_ljungbox(sarima.resid.iloc[8:], lags=[7, 14, 21], return_df=True)
R["ljung_box_p"] = {int(k): round(float(v), 4) for k, v in lb["lb_pvalue"].items()}

# Regression with SARIMA errors (SARIMAX): weather + calendar + trend as external regressors, d = D = 0.
# Assumes weather forecasts are available for the forecast window (we feed the observed weather).
SX_NAME = "SARIMAX (weather + trend)"
sx_aic = {}
for p, q, P, Q in itertools.product(range(3), range(2), range(2), range(2)):
    try:
        r_ = SARIMAX(tr_ts, exog=ex_tr, order=(p, 0, q), seasonal_order=(P, 0, Q, 7)).fit(disp=False, maxiter=100)
        sx_aic[(p, 0, q, P, 0, Q)] = r_.aic
    except Exception:
        pass
sxo = min(sx_aic, key=sx_aic.get)
sarx = SARIMAX(tr_ts, exog=ex_tr, order=sxo[:3], seasonal_order=(sxo[3], 0, sxo[5], 7)).fit(disp=False, maxiter=300)
record(SX_NAME, sarx.get_forecast(H, exog=ex_te).predicted_mean, {"aic": float(sarx.aic)})
R["sarimax_order"], R["sarimax_grid_n"] = list(sxo), len(sx_aic)
R["sarimax_coefs"] = {k: {"coef": round(float(sarx.params[k]), 3), "p": round(float(sarx.pvalues[k]), 4)} for k in exog_cols}
lb2 = acorr_ljungbox(sarx.resid.iloc[8:], lags=[7, 14, 21], return_df=True)
R["ljung_box_p_sarimax"] = {int(k): round(float(v), 4) for k, v in lb2["lb_pvalue"].items()}

# Exponential smoothing family
ses = ExponentialSmoothing(tr_ts, trend=None, seasonal=None).fit()
record("Simple exp. smoothing", ses.forecast(H), {"aic": float(ses.aic)})
holt = ExponentialSmoothing(tr_ts, trend="add", damped_trend=True, seasonal=None).fit()
record("Holt (damped trend)", holt.forecast(H), {"aic": float(holt.aic)})
hw = {}
for tr_, se_, dp_ in [("add", "add", False), ("add", "mul", False), ("add", "add", True), ("add", "mul", True)]:
    try:
        m_ = ExponentialSmoothing(tr_ts, trend=tr_, damped_trend=dp_, seasonal=se_, seasonal_periods=7).fit()
        hw[(tr_, se_, dp_)] = m_
    except Exception:
        pass
hk = min(hw, key=lambda k: hw[k].aic)
hwm = hw[hk]
hw_name = f"Holt-Winters ({hk[1]} seasonal{', damped' if hk[2] else ''})"
record(hw_name, hwm.forecast(H), {"aic": float(hwm.aic)})
R["hw_config"] = {"trend": hk[0], "seasonal": hk[1], "damped": hk[2],
                  "alpha": round(float(hwm.params["smoothing_level"]), 3),
                  "beta": round(float(hwm.params["smoothing_trend"]), 3),
                  "gamma": round(float(hwm.params["smoothing_seasonal"]), 3)}
R["ts_rows"] = rows
for r in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
best_ts = min([r for r in rows if r["model"] != SX_NAME], key=lambda r: r["rmse"])["model"]
R["best_ts_no_exog"] = best_ts

# forecast figure
fig, ax = plt.subplots(figsize=(10, 4.2))
hist = ts_raw.iloc[-120:]
ax.plot(hist.index, hist.values, color=GREY, lw=1.6, label="Actual (raw)")
show = [("Seasonal naive (t-7)", "#b8b8b8"), (R["sarima_name"], NAVY),
        (hw_name, TEAL), (SX_NAME, ORANGE)]
for n, c in show:
    ax.plot(fc[n].index, fc[n].values, color=c, lw=2, label=n, ls="-" if n != "Seasonal naive (t-7)" else ":")
ax.axvline(te_ts.index[0], color="k", ls=":", lw=1)
ax.text(te_ts.index[1], hist.max() * .97, "forecast period", fontsize=8)
ax.set(title="60-day hold-out forecasts of daily rentals", ylabel="Rentals per day")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(fontsize=8, loc="lower left", ncol=2)
savefig("fig15_forecast.png")

# residual diagnostics of the SARIMA model
res_s = sarima.resid.iloc[8:]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(res_s.index, res_s.values, color=NAVY, lw=.9)
ax[0].axhline(0, color=ORANGE, ls="--")
ax[0].set(title="SARIMA residuals over time")
ax[1].hist(res_s, bins=30, color=NAVY, edgecolor="white")
ax[1].set(title="Residual histogram")
plot_acf(res_s, lags=30, ax=ax[2], color=NAVY, title="Residual ACF")
savefig("fig16_residuals.png")

# rolling-origin (expanding window) evaluation, 30-day horizon, 4 folds
HZ, folds = 30, []
n = len(ts)
for end in [n - 90, n - 60, n - 30, n]:
    tr_f, te_f = ts_clean.iloc[:end - HZ], ts_raw.iloc[end - HZ:end]
    sn = np.tile(tr_f.iloc[-7:].values, 5)[:HZ]
    sa = SARIMAX(tr_f, order=so[:3], seasonal_order=(so[3], so[4], so[5], 7)).fit(disp=False, maxiter=100)
    hw_ = ExponentialSmoothing(tr_f, trend=hk[0], damped_trend=hk[2], seasonal=hk[1], seasonal_periods=7).fit()
    sx_ = SARIMAX(tr_f, exog=ex.iloc[:end - HZ], order=sxo[:3], seasonal_order=(sxo[3], 0, sxo[5], 7)).fit(disp=False, maxiter=200)
    folds.append({"origin": str(tr_f.index[-1].date()),
                  "naive7": rmse(te_f, sn), "sarima": rmse(te_f, sa.get_forecast(HZ).predicted_mean),
                  "hw": rmse(te_f, hw_.forecast(HZ)),
                  "sarimax": rmse(te_f, sx_.get_forecast(HZ, exog=ex.iloc[end - HZ:end]).predicted_mean)})
R["rolling"] = folds
R["rolling_mean"] = {k: float(np.mean([f[k] for f in folds])) for k in ["naive7", "sarima", "hw", "sarimax"]}
print("rolling", R["rolling_mean"])

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

with open(os.path.join(BASE, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=conv)
section("DONE - figures in ./figures, numbers in results.json")
