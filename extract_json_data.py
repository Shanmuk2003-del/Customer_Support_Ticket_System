
from pathlib import Path

import pandas as pd


def extract_cfpb_data():
    input_path = Path("data/raw/cfpb/complaints-2021-05-14_08_16_.json")
    output_path = Path("data/raw/cfpb/complaints_extracted.csv")

    # Check whether the input file exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"CFPB JSON dataset not found: {input_path}"
        )

    try:
        # Load original JSON
        df = pd.read_json(input_path)

        # Validate expected JSON structure
        if "_source" not in df.columns:
            raise ValueError(
                "Expected '_source' column was not found in the JSON dataset."
            )

        # Extract nested data
        data = pd.json_normalize(df["_source"])

        if data.empty:
            raise ValueError("Extracted dataset is empty.")

        # Display dataset information
        print("\n--- EXTRACTED DATASET SHAPE ---")
        print(data.shape)

        print("\n--- EXTRACTED COLUMN NAMES ---")
        print(data.columns.tolist())

        print("\n--- DATA TYPES ---")
        print(data.dtypes)

        print("\n--- MISSING VALUES ---")
        print(data.isnull().sum().sort_values(ascending=False))

        # Create output directory if necessary
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save extracted data
        data.to_csv(output_path, index=False)

        print(f"\nExtracted dataset saved successfully: {output_path}")

    except (ValueError, KeyError, OSError) as error:
        print(f"\nDataset extraction failed: {error}")
        raise


if __name__ == "__main__":
    extract_cfpb_data()
