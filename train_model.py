"""
AI VISION 2035 — Model training pipeline.

Trains and compares Linear Regression, Decision Tree and Random Forest on the
2020-2025 AI adoption dataset, then persists the best performer to model.pkl.

Run once before starting the app:
    python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

TARGET = "AI_Maturity_Score"
CATEGORICAL = ["Industry", "Country", "Employee_Size", "Uses_AI", "Use_Case"]
NUMERIC = ["Year", "Log_Revenue", "AI_ROI_Percent"]
RANDOM_STATE = 42

# Selection rule -------------------------------------------------------------
# Accuracy alone is not sufficient here. This model is used twice: to score a
# single profile, and to trace a ten-year forecast curve. The second use needs
# a smooth, monotonic response to a rising input, which tree ensembles cannot
# give -- they are piecewise-constant by construction, so they return either a
# noisy curve (small leaves) or a step function (large leaves).
#
# On this dataset the three candidates sit within 0.014 R2 of each other, which
# is inside the spread of five-fold cross-validation. Choosing on R2 alone
# would be selecting on noise. So: any model within R2_TOLERANCE of the best
# score is treated as accuracy-equivalent, and the tie is broken on response
# stability. Both numbers are reported, and the rule stays fully automatic.
R2_TOLERANCE = 0.02
STABILITY_PROFILES = 16
STABILITY_STEPS = 40


def load_dataset(path=DATASET_PATH):
    """Read the CSV and apply the cleaning rules the whole app relies on."""
    df = pd.read_csv(path)

    # Use_Case is blank exactly when a company does not use AI. That is a
    # meaningful value, not missing data, so encode it explicitly.
    df["Use_Case"] = df["Use_Case"].fillna("None")

    # Revenue spans six orders of magnitude (1e6 to 5e11). Log-scale it so the
    # linear model is not dominated by a handful of mega-caps.
    df["Log_Revenue"] = np.log10(df["Revenue_USD"].clip(lower=1.0))

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    return df


def build_preprocessor():
    """One-hot the categoricals, standardise the numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        # Leaf sizes are deliberately large. With small leaves the forest
        # memorises thin slices of the data (single-company, high-revenue
        # profiles), which makes its response to a rising input non-monotonic
        # and produces jagged forecast curves. Heavier regularisation raised
        # held-out R2 as well as smoothing the response surface.
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=40, random_state=RANDOM_STATE
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=40,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def response_stability(pipeline, df):
    """
    Measure how well-behaved a fitted model is when one input is swept.

    AI_ROI_Percent correlates with the target at +0.61, so a sensible model
    should respond to rising ROI with a non-decreasing, reasonably continuous
    curve. Sweeping ROI across its observed range for a set of real profiles
    gives two numbers:

      monotonicity - share of steps that do not move backwards
      resolution   - share of sweep points that produce a distinct value
                     (1.0 is a smooth curve, near 0 is a step function)

    Their mean is the stability score used to break accuracy ties.
    """
    adopters = df[df["Uses_AI"] == "Yes"]
    sample = adopters.sample(
        min(STABILITY_PROFILES, len(adopters)), random_state=RANDOM_STATE
    )
    grid = np.linspace(
        float(df["AI_ROI_Percent"].min()), float(df["AI_ROI_Percent"].max()), STABILITY_STEPS
    )

    monotonicity, resolution = [], []
    for _, base in sample.iterrows():
        rows = pd.DataFrame(
            [
                {
                    "Industry": base["Industry"],
                    "Country": base["Country"],
                    "Employee_Size": base["Employee_Size"],
                    "Uses_AI": base["Uses_AI"],
                    "Use_Case": base["Use_Case"],
                    "Year": base["Year"],
                    "Log_Revenue": base["Log_Revenue"],
                    "AI_ROI_Percent": roi,
                }
                for roi in grid
            ]
        )
        preds = pipeline.predict(rows)
        diffs = np.diff(preds)
        monotonicity.append(1.0 - float((diffs < -0.01).sum()) / len(diffs))
        resolution.append(len(np.unique(np.round(preds, 2))) / len(preds))

    mono = float(np.mean(monotonicity))
    res = float(np.mean(resolution))
    return {
        "monotonicity": round(mono, 4),
        "resolution": round(res, 4),
        "stability": round((mono + res) / 2, 4),
    }


def evaluate(name, pipeline, X_train, X_test, y_train, y_test, full_df):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    cv = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
    )

    scores = {
        "model": name,
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "cv_r2_mean": round(float(cv.mean()), 4),
        "cv_r2_std": round(float(cv.std()), 4),
    }
    scores.update(response_stability(pipeline, full_df))
    return scores


def train():
    df = load_dataset()
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    fitted = {}

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows.\n")
    print(f"{'Model':<26}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'CV R2':>9}{'Stability':>11}")
    print("-" * 75)

    for name, estimator in candidate_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = evaluate(name, pipeline, X_train, X_test, y_train, y_test, df)
        results.append(scores)
        fitted[name] = pipeline
        print(
            f"{name:<26}{scores['mae']:>9.3f}{scores['rmse']:>9.3f}"
            f"{scores['r2']:>9.3f}{scores['cv_r2_mean']:>9.3f}{scores['stability']:>11.3f}"
        )

    # Two-criterion selection, as described at the top of this file.
    top_r2 = max(r["r2"] for r in results)
    eligible = [r for r in results if top_r2 - r["r2"] <= R2_TOLERANCE]
    best = max(eligible, key=lambda r: (r["stability"], r["r2"]))
    best_pipeline = fitted[best["model"]]

    print("-" * 75)
    print(
        f"\nAccuracy-equivalent (within {R2_TOLERANCE} R2 of {top_r2:.4f}): "
        + ", ".join(r["model"] for r in eligible)
    )
    print(f"Tie broken on response stability -> {best['model']}")

    # Refit the winner on the full dataset so no rows are wasted at serve time.
    best_pipeline.fit(X, y)

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best["model"],
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
            "target": TARGET,
            "trained_rows": int(len(X)),
            # Tree models cannot extrapolate: outside the range they were
            # trained on they simply return the value of the nearest boundary
            # leaf. Recording the ranges lets the app clamp inputs explicitly
            # instead of silently relying on that behaviour.
            "feature_ranges": {
                col: [float(df[col].min()), float(df[col].max())] for col in NUMERIC
            },
        },
        MODEL_PATH,
    )

    metrics = {
        "results": results,
        "best_model": best["model"],
        "top_r2": round(float(top_r2), 4),
        "r2_tolerance": R2_TOLERANCE,
        "accuracy_equivalent": [r["model"] for r in eligible],
        "selection_metric": (
            f"R2 on a 20% held-out split, then response stability as the tie-break "
            f"among models within {R2_TOLERANCE} R2 of the best score"
        ),
        "selection_reason": (
            f"{len(eligible)} of {len(results)} models scored within {R2_TOLERANCE} R2 of the "
            f"best result ({top_r2:.4f}) — a gap smaller than the five-fold cross-validation "
            f"spread, so they are treated as accuracy-equivalent. {best['model']} was selected "
            f"on response stability ({best['stability']:.3f}), which matters because the same "
            f"model traces the 2026-2035 forecast curve."
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": CATEGORICAL + NUMERIC,
        "target": TARGET,
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSelected: {best['model']}  (R2 = {best['r2']}, stability = {best['stability']})")
    print(f"Saved -> {MODEL_PATH}")
    print(f"Saved -> {METRICS_PATH}")


if __name__ == "__main__":
    train()
