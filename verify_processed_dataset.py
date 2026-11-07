
import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "data/processed/cfpb_model_ready.csv"
)

print("\n--- DATASET VERIFICATION ---")

# 1. Shape
print("Dataset shape:", df.shape)

# 2. Columns
print("\nColumns:")
print(df.columns.tolist())

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Empty complaint descriptions
print("\nEmpty descriptions:")
print(
    (df["complaint_what_happened"].str.strip() == "").sum()
)

# 5. Duplicate complaint IDs
print("\nDuplicate complaint IDs:")
print(df["complaint_id"].duplicated().sum())

# 6. Duplicate descriptions
print("\nDuplicate descriptions:")
print(
    df["complaint_what_happened"].duplicated().sum()
)

# 7. Category distribution
print("\nCategory distribution:")
print(df["product"].value_counts())

# 8. Check conflicting labels
conflicts = (
    df.groupby("complaint_what_happened")["product"]
    .nunique()
)

print("\nRemaining conflicting descriptions:")
print((conflicts > 1).sum())

print("\nDataset verification completed.")
