
import pandas as pd
from pathlib import Path

# Load extracted dataset
INPUT_PATH = "data/raw/cfpb/complaints_extracted.csv"
df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

# Validate required input columns
REQUIRED_COLUMNS = ["complaint_id", "complaint_what_happened", "product"]
missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}. "
        f"Expected columns: {REQUIRED_COLUMNS}. "
        f"Found columns: {df.columns.tolist()}"
    )

# 1. Clean complaint text
df["complaint_what_happened"] = (
    df["complaint_what_happened"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# 2. Remove empty descriptions
df = df[
    df["complaint_what_happened"] != ""
].copy()

print("Records with complaint text:", len(df))

# 3. Keep categories with at least 500 records
category_counts = df["product"].value_counts()

selected_categories = category_counts[
    category_counts >= 500
].index.tolist()

df = df[
    df["product"].isin(selected_categories)
].copy()

print("\nSelected categories:")
print(df["product"].value_counts())

# 4. Identify descriptions with conflicting labels
text_label_counts = (
    df.groupby("complaint_what_happened")["product"]
    .nunique()
)

conflicting_texts = text_label_counts[
    text_label_counts > 1
].index

print(
    "\nConflicting descriptions found:",
    len(conflicting_texts)
)

# 5. Remove all records containing conflicting descriptions
records_before_conflict_removal = len(df)
df = df[
    ~df["complaint_what_happened"].isin(conflicting_texts)
].copy()
records_removed_conflict = records_before_conflict_removal - len(df)

print(
    f"\nRecords removed due to conflicting labels: {records_removed_conflict}"
)

# 6. Keep only required columns
final_df = df[
    [
        "complaint_id",
        "complaint_what_happened",
        "product"
    ]
].copy()

# 7. Save processed dataset
output_path = Path("data/processed/cfpb_model_ready.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(output_path, index=False)

print("\n--- FINAL DATASET SUMMARY ---")
print("Final records:", len(final_df))
print("Final categories:", final_df["product"].nunique())
print("Duplicate complaint IDs:", final_df["complaint_id"].duplicated().sum())

print("\nFinal category distribution:")
print(final_df["product"].value_counts())

print("\nSaved successfully to:", output_path)
