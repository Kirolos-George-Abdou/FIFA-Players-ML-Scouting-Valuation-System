#  FIFA Players — ML Scouting & Valuation System
 
> A unified machine learning pipeline that predicts player market value (regression) and classifies players into performance tiers (classification) using an ensemble of 5+ models.
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.x-green)
 
---
 
##  Project Overview
 
This project builds a **dual-task ML system** on FIFA player data:
 
- **Regression** — Predict player market value (`Value Per M$`)
- **Classification** — Assign players to performance tiers: `Low / Mid / High / Elite`
Both tasks are solved using multiple models, tuned with cross-validation, and combined into ensemble systems for maximum predictive power.
 
---
 
##  Dataset
 
**Source:** `Fifa.csv`
 
| Property | Value |
|----------|-------|
| Shape | ~18,000+ players, 13 features |
| Target (Regression) | `Value Per M$` — continuous, heavily right-skewed |
| Target (Classification) | Performance tier derived from `Overall_Rating` quartiles |
| Categorical features | Country, Position, Team |
| Numerical features | Age, Future Potential, Total_Stats Score, Overall_Rating |
 
---
 
##  Preprocessing Pipeline
 
1. **EDA** — histograms, correlation heatmap, skewness analysis, average rating per position
2. **Train/test split** — 80/20 before any preprocessing (prevents data leakage)
3. **Log transform** — applied to target (`np.log1p`) and skewed numeric features
4. **Outlier handling** — extreme values retained (elite players are real); log transform compresses their influence
5. **Encoding**:
   - Regression → `TargetEncoder` for Country, Position, Team
   - Classification → `OneHotEncoder` for Country, Position, Team
6. **Scaling** — `StandardScaler` on numeric features (Age, Future Potential, Total_Stats Score, Overall_Rating)
7. **Leakage prevention** — `Overall_Rating` dropped from all classifier inputs (tiers are derived from it)
---
 
##  Classification Target
 
Players binned into 4 tiers using `Overall_Rating` quartiles:
 
| Tier | Threshold |
|------|-----------|
| Low | < 25th percentile |
| Mid | < 50th percentile |
| High | < 75th percentile |
| Elite | ≥ 75th percentile |
 
---
 
##  Models Trained
 
### Regression (predicting `Value Per M$`)
 
| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Polynomial Regression (degree 1–4) | Degree 3 = best generalization |
| Ridge Regression | Best regularized model; alpha swept via log-space |
| Lasso Regression | Feature selection — zeroed out 11 features |
| KNN Regressor | Tuned via GridSearchCV |
| SVR (RBF kernel) | Tuned via GridSearchCV |
| Random Forest Regressor | Tuned via HalvingGridSearchCV |
| **Voting Ensemble** | Ridge + Lasso + KNN + SVR + RF |
 
### Classification (predicting performance tier)
 
| Model | Notes |
|-------|-------|
| Logistic Regression (L1 + L2) | Best single classifier (~81.5% accuracy) |
| GaussianNB | 3 numeric features only |
| BernoulliNB | Binarized OHE features |
| ComplementNB | MinMax-scaled features |
| KNN Classifier | Tuned via GridSearchCV |
| SVM (RBF kernel) | Tuned via GridSearchCV |
| Random Forest Classifier | Tuned via HalvingGridSearchCV |
| **Stacking Ensemble** | LR + NB + KNN + SVM + RF → LR meta-learner |
 
---
 
##  Results
 
### Classification
 
| Model | Accuracy |
|-------|:--------:|
| GaussianNB | ~70.3% |
| Logistic Regression (L2) | ~81.5% |
| KNN Classifier (optimized) | ~81.4% |
| SVM (RBF) | ~84.2% |
| Random Forest | ~85%+ |
| **Stacking Ensemble** | **~85%+** |
 
### Regression
 
| Model | Test R² |
|-------|:-------:|
| Linear Regression (baseline) | Low |
| Ridge (degree=3, best α) | Moderate |
| KNN Regressor (optimized) | Moderate |
| **Voting Ensemble** | **~97.8%** |
 
---
 
##  Ensemble System
 
**Classification — Stacking:**
Five base classifiers (LR, NB, KNN, SVM, RF) feed predictions into a Logistic Regression meta-learner trained via 5-fold CV.
 
**Regression — Voting:**
Five regressors (Ridge, Lasso, KNN, SVR, RF) each predict on log-scale; their outputs are averaged.
 
**Stability (5-fold CV):**
- Classification: mean accuracy with low ±std across folds
- Regression: mean R² ≈ 0.97+ with low ±std across folds
---

##  Inference Pipeline
 
The final system randomly selects players from the test set and outputs:
 
```
PLAYER: LIONEL MESSI
info: 34 years old | RW | Paris Saint-Germain
stats: overall 93 | potential 95
prediction: $112.50M valuation | Elite tier
```
 
---
 
##  Visualizations
 
- Feature histograms and Pearson correlation heatmap
- Target skewness and log-transform effect
- IQR box plots before and after log transform
- Polynomial degree vs R² curve
- Ridge/Lasso alpha sweep (train vs test RMSE)
- Logistic Regression C sweep (L1 vs L2)
- Naive Bayes variant accuracy comparison
- KNN k-value sweep (train vs test accuracy/R²)
- GridSearchCV heatmaps (k × metric × weights)
- Learning curves for RF classifier and regressor
- Feature importance charts
- Confusion matrices for all classifiers
- Stratified K-Fold bar charts (fold-by-fold accuracy)
---
 
##  Getting Started
 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```
 
Place `Fifa.csv` in the project root, then run the notebook top to bottom.
 
---
 
##  Dependencies
 
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```
 
---
 
##  Key Insights
 
**Why Ridge outperforms Lasso here:** OHE-encoded categorical columns (Country, Position, Team) produce many binary features that are individually weak but collectively informative. Ridge shrinks all coefficients smoothly; Lasso eliminates entire groups, discarding collective signal.
 
**Why Logistic Regression beats Naive Bayes:** LR uses the full encoded feature matrix; GaussianNB is limited to 3 numeric features, capping its accuracy ceiling.
 
**Why degree 3 is optimal:** Degree 1–2 underfit the non-linear value relationship; degree 4 causes a widening train/test gap (overfitting).
 
**Why ensemble > single models:** Stacking and voting reduce individual model variance and combine complementary inductive biases across algorithms.
 
---
 
##  Authors
 
Built as a Machine Learning course project.
 
---
 
##  License
 
This project is licensed under the MIT License.
