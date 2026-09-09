# AI VISION 2035

**Global AI Adoption Across Industries & Future Forecasting Platform**

A Flask analytics application built over 6,000 company-year records of AI
adoption (2020–2025). It summarises what the history shows and projects AI
maturity scores to 2035 using a trained regression model.

---

## Running it

```bash
pip install -r requirements.txt
python train_model.py     # trains 3 models, saves the winner to model.pkl
python app.py             # http://127.0.0.1:5000
```

`train_model.py` must run once before `app.py`. It writes `model.pkl` and
`model_metrics.json`; the app reads both at startup.

---

## Project structure

```
AI_VISION_2035/
├── app.py                  Flask routes + JSON API
├── train_model.py          Training, comparison, selection, persistence
├── model.pkl               Saved pipeline (Joblib)
├── model_metrics.json      Per-model MAE / RMSE / R² / CV R² / stability
├── dataset.csv             Source data (6,000 rows × 11 columns)
├── requirements.txt
├── templates/
│   ├── base.html           Shared layout (nav, background, footer)
│   ├── index.html          Home
│   ├── about.html          About
│   └── dashboard.html      Dashboard
└── static/
    ├── css/                base · pages · charts · dashboard
    ├── js/                 charts.js · main.js · dashboard.js
    ├── icons/              favicon.svg
    └── images/
```

`base.html` holds the markup shared by all three pages. The three pages the
brief asks for are `index`, `about` and `dashboard`.

---

## API

| Method | Endpoint          | Returns                                             |
| ------ | ----------------- | --------------------------------------------------- |
| GET    | `/api/summary`    | Summary-card figures and thin-segment flags          |
| GET    | `/api/historical` | All eight chart series plus generated findings       |
| GET    | `/api/options`    | Dropdown values and scenario definitions             |
| GET    | `/api/model`      | Metrics for all three models and the selection rule  |
| POST   | `/api/forecast`   | Prediction, trajectory, assumptions, recommendation  |

`POST /api/forecast` body:

```json
{ "company": "Amazon", "country": "", "industry": "", "year": 2030, "scenario": "moderate" }
```

All fields are optional except `year`. Omitting `company` forecasts a segment
profile built from country/industry medians.

---

## Machine learning

**Target:** `AI_Maturity_Score` (continuous, 0–100) — a regression problem.

**Features (8):** `Year`, `Industry`, `Country`, `Employee_Size`, `Uses_AI`,
`Use_Case`, `Revenue_USD` (log₁₀), `AI_ROI_Percent`.

`Company` is deliberately excluded. With 1,000 identities at six rows each,
one-hot encoding it would let the model memorise individual firms instead of
learning transferable structure, and it would be useless for any company not
already in the file.

**Preprocessing:** one-hot for categoricals, standardisation for numerics, all
inside a scikit-learn `Pipeline` so it is fitted within each CV fold rather
than on the full dataset. Blank `Use_Case` values become an explicit `None`
level — they mark non-adopters, not missing data.

### Results

| Model             | MAE    | RMSE   | R²    | CV R²         | Stability |
| ----------------- | ------ | ------ | ----- | ------------- | --------- |
| Linear Regression | 11.444 | 13.223 | 0.622 | 0.623 ± 0.010 | **0.919** |
| Decision Tree     | 11.277 | 13.086 | 0.630 | 0.626         | 0.512     |
| Random Forest     | 11.206 | 12.977 | **0.636** | 0.634     | 0.659     |

### Why selection is not on R² alone

The three models sit within **0.014 R²** of each other — smaller than the
five-fold cross-validation spread. Choosing the top number would be selecting
on noise.

The model is also used twice: to score one profile, and to trace a ten-year
forecast curve. The second use needs a smooth, non-decreasing response to a
rising input, which tree ensembles cannot give — they are piecewise-constant by
construction. Tuned with small leaves the Random Forest produced a jagged
curve (`88.2 → 87.5 → 84.5 → 84.8 → 86.3`); tuned with large leaves it produced
a flat line with cliffs (`85.8 × 4, then 69.6, 67.8, 52.1`). Neither is a
defensible forecast.

So selection has two automatic stages:

1. Any model within `R2_TOLERANCE = 0.02` of the best score is treated as
   accuracy-equivalent.
2. The tie is broken on **response stability** — `AI_ROI_Percent` is swept
   across its observed range for 16 real profiles, measuring the share of steps
   that do not move backwards and the share of sweep points giving a distinct
   value.

Nothing is hard-coded: re-run `train_model.py` on different data and the rule
may pick a different model.

---

## Forecasting

The model maps features to a score; it does not extrapolate time. To forecast a
future year, the input features are projected forward first, then passed
through the saved pipeline. That projection is where the assumptions live, so
the dashboard makes you pick one:

| Scenario              | ROI drift    | Revenue growth | Basis              |
| --------------------- | ------------ | -------------- | ------------------ |
| Data trend            | segment slope | segment CAGR   | measured from data |
| Moderate adoption     | +1.5 pts/yr  | +4 %/yr        | stated assumption  |
| Accelerated adoption  | +3.5 pts/yr  | +8 %/yr        | stated assumption  |

Every numeric input is clamped to the range the model was trained on. A tree
asked about 2035 has no split beyond 2025 and returns its boundary leaf
regardless; clamping makes that explicit and stops projected revenue drifting
into a region the model has never scored. Predictions are clipped to 0–100.

Growth is measured against a 2025 baseline scored by the same model on the same
profile, so the comparison isolates the effect of the projected features rather
than folding model error into the difference.

---

## Two things about this dataset

**There is no time trend.** Mean maturity moves 55.0 (2020) → 54.7 (2025), and
`Year` correlates with the target at roughly zero. A forecast here cannot be
extrapolation from history, which is why projections are driven by explicit
scenario assumptions. The "Data trend" scenario produces a near-flat line, and
that is the honest baseline.

**980 of 1,000 companies are generated,** and their `Industry` and `Country` are
re-randomised every year — only 2 % of companies keep a stable industry. This is
why South Korea (n=6) and Industrial (n=6) top the rankings: they are Samsung
and one other real firm. Any segment under 60 records is drawn in amber, labelled
in its tooltip, and excluded from the generated findings.

The strongest real signal is the adoption gap: companies with an active AI use
case average **62.8** against **19.7** for those without — a 43-point difference
that dwarfs every other variable. The full spread between the best and worst
well-sampled country is under three points.

---

## Technology

HTML5 · CSS3 · JavaScript (ES6) · Python 3 · Flask · Pandas · NumPy ·
scikit-learn · Joblib

No React, Bootstrap, Tailwind, Node, Django, PHP, MySQL, MongoDB, Firebase, CSS
framework, JS framework, CDN or external API.

`static/js/charts.js` is a hand-written SVG chart engine (~330 lines) covering
line, area, bar, scatter and donut, with tooltips, entry animation, legends and
`ResizeObserver`-driven redraw — written from scratch because no chart library
was permitted.

Accessibility and polish: responsive to mobile, visible keyboard focus rings,
and `prefers-reduced-motion` respected throughout (the neural canvas is disabled
and all transitions collapse).
