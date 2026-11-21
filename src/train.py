"""
Loan Approval Prediction - training pipeline.

Steps: load -> clean -> feature engineering -> preprocessing pipeline ->
compare models with cross-validation -> tune best -> evaluate on hold-out
test set -> save model, metrics and plots.

Usage:  python src/train.py [--data data/loan_data.csv]
"""
import argparse, json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
FIG = "reports/figures/"

CAT = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
       "Property_Area"]
NUM = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
       "Loan_Amount_Term", "Credit_History", "TotalIncome", "EMI",
       "EMI_to_Income", "LogTotalIncome", "LogLoanAmount"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering - also used by the web app at prediction time."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    # LoanAmount is in thousands, so EMI is in rupees per month (simple, no interest)
    df["EMI"] = df["LoanAmount"] * 1000 / df["Loan_Amount_Term"]
    df["EMI_to_Income"] = df["EMI"] / df["TotalIncome"].replace(0, np.nan)
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    return df


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


def make_pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def eda_plots(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df["Loan_Status"].value_counts().plot.bar(ax=ax[0], color=["#2e7d32", "#c62828"])
    ax[0].set_title("Class distribution (Y = approved)")
    pd.crosstab(df["Credit_History"], df["Loan_Status"], normalize="index") \
        .plot.bar(stacked=True, ax=ax[1], color=["#c62828", "#2e7d32"])
    ax[1].set_title("Approval rate by credit history")
    sns.boxplot(data=df, x="Loan_Status", y="LogTotalIncome", ax=ax[2])
    ax[2].set_title("Log total income vs. outcome")
    plt.tight_layout(); plt.savefig(FIG + "01_eda.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[NUM + ["Target"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Correlation matrix"); plt.tight_layout()
    plt.savefig(FIG + "02_correlation.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    df.isna().sum().loc[lambda s: s > 0].sort_values().plot.barh(color="#1565c0")
    plt.title("Missing values per column"); plt.tight_layout()
    plt.savefig(FIG + "00_missing.png", dpi=150); plt.close()


def main(path: str):
    raw = pd.read_csv(path)
    df = add_features(raw)
    df["Target"] = (df["Loan_Status"] == "Y").astype(int)
    print(f"Loaded {len(df)} rows | approval rate {df.Target.mean():.1%}")
    eda_plots(df)

    y = df["Target"]
    X = df[CAT + NUM]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                 eval_metric="logloss", random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows = []
    for name, m in models.items():
        auc = cross_val_score(make_pipe(m), X_tr, y_tr, cv=cv, scoring="roc_auc")
        acc = cross_val_score(make_pipe(m), X_tr, y_tr, cv=cv, scoring="accuracy")
        rows.append({"Model": name, "CV_Accuracy": acc.mean(), "CV_AUC": auc.mean(),
                     "CV_AUC_std": auc.std()})
        print(f"{name:20s} CV acc {acc.mean():.3f} | CV AUC {auc.mean():.3f} +/- {auc.std():.3f}")
    cmp_df = pd.DataFrame(rows).sort_values("CV_AUC", ascending=False)
    cmp_df.to_csv("reports/model_comparison.csv", index=False)

    # Tune the best candidate family (Gradient Boosting vs XGBoost are usually top)
    best_name = cmp_df.iloc[0]["Model"]
    print("\nBest by CV AUC:", best_name, "-> tuning")
    grids = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"),
                                {"model__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"model__max_depth": [3, 4, 5, 7], "model__min_samples_leaf": [5, 20, 50]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED),
                          {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 10],
                           "model__min_samples_leaf": [2, 5, 10]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1],
                               "model__max_depth": [2, 3]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED),
                    {"model__n_estimators": [150, 300], "model__learning_rate": [0.03, 0.1],
                     "model__max_depth": [2, 3, 4]}),
    }
    base, grid = grids[best_name]
    gs = GridSearchCV(make_pipe(base), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # Final evaluation on untouched test set
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "best_model": best_name,
        "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                        for k, v in gs.best_params_.items()},
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    print("\nTEST METRICS:", json.dumps(metrics, indent=2, default=float))

    ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                           display_labels=["Rejected", "Approved"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
    plt.savefig(FIG + "03_confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for name, m in models.items():
        p = make_pipe(m).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG + "04_roc.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    cmp_df.set_index("Model")["CV_AUC"].sort_values().plot.barh(color="#1565c0")
    plt.xlim(0.5, 1); plt.title("Model comparison - 5-fold CV ROC-AUC")
    plt.tight_layout(); plt.savefig(FIG + "05_model_comparison.png", dpi=150); plt.close()

    # Feature importance (model-agnostic permutation importance)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc",
                                n_repeats=15, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    imp.tail(10).plot.barh(figsize=(7, 5), color="#2e7d32")
    plt.title("Permutation importance (drop in ROC-AUC)"); plt.tight_layout()
    plt.savefig(FIG + "06_feature_importance.png", dpi=150); plt.close()
    imp.sort_values(ascending=False).to_csv("reports/feature_importance.csv")

    joblib.dump(best, "models/loan_model.joblib")
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2, default=float)
    print("\nSaved model, metrics and figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/loan_data.csv")
    main(ap.parse_args().data)
