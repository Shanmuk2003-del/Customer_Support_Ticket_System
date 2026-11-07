"""
split_dataset.py

Creates a leakage-safe train/test split for the CFPB complaint dataset.

WHY THIS MATTERS:
Some complaints have identical text (128 repeated descriptions in this dataset).
If the same description appears in BOTH training and testing, the model can
"memorize" the answer during training and appear more accurate than it really is
when evaluated. This is called data leakage.

To prevent this, we group identical descriptions together so they ALWAYS stay
in the same split (either all in training or all in testing, never separated).

We use StratifiedGroupKFold:
  - "Stratified" keeps the product-category distribution similar in train and test.
  - "Group"      ensures identical descriptions are never separated into different splits.
  - 5 folds      gives us 5 candidate splits; we pick the one closest to 80/20.
"""

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Safety check: StratifiedGroupKFold requires scikit-learn >= 1.4.
# If it is unavailable, STOP and explain instead of silently falling back
# to an ordinary random split (which would cause data leakage).
# ---------------------------------------------------------------------------
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:
    print("ERROR: StratifiedGroupKFold is not available in your scikit-learn installation.")
    print("Please upgrade scikit-learn to version 1.4 or newer:")
    print("    pip install --upgrade scikit-learn")
    print(f"Original error: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load the processed dataset
# ---------------------------------------------------------------------------
INPUT_PATH = "data/processed/cfpb_model_ready.csv"
df = pd.read_csv(INPUT_PATH)

print(f"Loaded dataset: {INPUT_PATH}")
print(f"Total records : {len(df)}")
print(f"Columns       : {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. Define features (X), target (y), and grouping variable
# ---------------------------------------------------------------------------
# X = the complaint text (what happened)
# y = the product category (what we want to predict)
# groups = the complaint text again, used to keep identical descriptions together
# ---------------------------------------------------------------------------
X = df["complaint_what_happened"]
y = df["product"]
groups = df["complaint_what_happened"]

# ---------------------------------------------------------------------------
# 3. Split using StratifiedGroupKFold (5 folds, shuffled, fixed seed)
# ---------------------------------------------------------------------------
# We want approximately 80% training / 20% testing.
# With 5 folds, each fold's test set is roughly 20% of the data.
# We iterate through all folds and pick the one whose test size is closest
# to the ideal 20% target.
# ---------------------------------------------------------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

total = len(df)
target_test_size = int(round(total * 0.2))

best_fold = None
best_test_size = -1
best_train_idx = None
best_test_idx = None

print("\nFold sizes (each fold's test set is held out once):")
for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
    test_size = len(test_idx)
    test_ratio = test_size / total
    print(f"  Fold {fold_id}: train={len(train_idx)}, test={test_size}, "
          f"test_ratio={test_ratio:.4f}")

    # Pick the fold whose test size is closest to the 20% target
    if best_test_size == -1 or abs(test_size - target_test_size) < abs(best_test_size - target_test_size):
        best_test_size = test_size
        best_fold = fold_id
        best_train_idx = train_idx
        best_test_idx = test_idx

print(f"\nSelected fold {best_fold} (test ratio = {best_test_size / total:.4f})")

# ---------------------------------------------------------------------------
# 4. Build train and test DataFrames from the selected fold
# ---------------------------------------------------------------------------
train_df = df.iloc[best_train_idx].copy()
test_df = df.iloc[best_test_idx].copy()

# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)
print(f"Training records: {len(train_df)}")
print(f"Testing records : {len(test_df)}")
print(f"Total           : {len(train_df) + len(test_df)}")

print("\nCategory distribution - TRAINING:")
print(train_df["product"].value_counts().to_string())

print("\nCategory distribution - TESTING:")
print(test_df["product"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 7. Verify no leakage: check for identical descriptions shared between train/test
# ---------------------------------------------------------------------------
train_descriptions = set(train_df["complaint_what_happened"].unique())
test_descriptions = set(test_df["complaint_what_happened"].unique())
shared_descriptions = train_descriptions & test_descriptions

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)
print(f"Unique descriptions in training: {len(train_descriptions)}")
print(f"Unique descriptions in testing : {len(test_descriptions)}")
print(f"Identical descriptions shared between train and test: {len(shared_descriptions)}")

if len(shared_descriptions) == 0:
    print("\nOVERLAP CONFIRMED: ZERO shared descriptions. The split is leakage-safe.")
else:
    print(f"\nLEAKAGE DETECTED: {len(shared_descriptions)} descriptions appear in both datasets!")
    for desc in list(shared_descriptions)[:5]:
        print(f"  - {desc[:80]}...")