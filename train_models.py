"""
train_models.py

Trains two text-classification models for the CFPB complaint dataset:
  1. Logistic Regression
  2. Linear SVM Support Vector Machine (SVM)

Each model is wrapped in a Scikit-learn Pipeline:
    TfidfVectorizer -> classifier

The TF-IDF vectorizer is fitted ONLY on the training data so that the test
set never influences feature extraction (prevents data leakage).

Outputs:
  - Evaluation metrics for both models (accuracy, precision, recall, F1)
  - Classification reports
  - Confusion matrices
  - Saved fitted pipelines (joblib)
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dependency check: fail with a clear message instead of silently installing.
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    import joblib
except ImportError as exc:
    print("ERROR: A required dependency is missing.")
    print("Please install the missing package, for example:")
    print("    pip install scikit-learn joblib")
    print(f"Original error: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load the train/test splits created by split_dataset.py
# ---------------------------------------------------------------------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["complaint_what_happened"]
y_train = train_df["product"]
X_test = test_df["complaint_what_happened"]
y_test = test_df["product"]

print(f"Loaded training data: {len(X_train)} records")
print(f"Loaded testing data : {len(X_test)} records")

# ---------------------------------------------------------------------------
# 2. TF-IDF vectorizer configuration
#    - stop_words="english": removes common English words (the, and, is, ...)
#    - ngram_range=(1, 2): considers both single words and two-word phrases
#    - min_df=2: ignores words that appear in fewer than 2 documents
#    - max_df=0.95: ignores words that appear in more than 95% of documents
#    - sublinear_tf: scales down the raw term frequency so very frequent
#      words do not dominate the vector representation
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

# ---------------------------------------------------------------------------
# 3. Define the two models as Pipelines.
#    The vectorizer is fitted ONLY on training data inside the pipeline.
# ---------------------------------------------------------------------------
lr_pipeline = Pipeline(
    [
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

svm_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
    max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(class_weight="balanced")),
    ]
)

# ---------------------------------------------------------------------------
# 4. Train both models
# ---------------------------------------------------------------------------
print("\nTraining Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Linear SVM...")
svm_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate both models on the test set
# ---------------------------------------------------------------------------
def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)    : {rec:.4f}")
    print(f"Macro F1-score   : {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

lr_metrics = evaluate(lr_pipeline, "Logistic Regression", X_test, y_test)
svm_metrics = evaluate(svm_pipeline, "Linear SVM", X_test, y_test)

# ---------------------------------------------------------------------------
# 6. Save the fitted pipelines using joblib
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(lr_pipeline, "models/logistic_regression_pipeline.joblib")
joblib.dump(svm_pipeline, "models/linear_svm_pipeline.joblib")

print("\nSaved fitted pipelines:")
print("  - models/logistic_regression_pipeline.joblib")
print("  - models/linear_svm_pipeline.joblib")

# ---------------------------------------------------------------------------
# 7. Side-by-side comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Metric':<22}{'Logistic Regression':<22}{'Linear SVM':<22}")
print("-" * 66)
print(f"{'Accuracy':<22}{lr_metrics['accuracy']:<22.4f}{svm_metrics['accuracy']:<22.4f}")
print(f"{'Precision (macro)':<22}{lr_metrics['precision']:<22.4f}{svm_metrics['precision']:<22.4f}")
print(f"{'Recall (macro)':<22}{lr_metrics['recall']:<22.4f}{svm_metrics['recall']:<22.4f}")
print(f"{'Macro F1':<22}{lr_metrics['macro_f1']:<22.4f}{svm_metrics['macro_f1']:<22.4f}")
print(f"{'Weighted F1':<22}{lr_metrics['weighted_f1']:<22.4f}{svm_metrics['weighted_f1']:<22.4f}")

best_macro_f1 = (
    "Logistic Regression"
    if lr_metrics["macro_f1"] > svm_metrics["macro_f1"]
    else "Linear SVM"
)
print(f"\nBest model by Macro F1: {best_macro_f1}")