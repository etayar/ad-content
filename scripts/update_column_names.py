import pandas as pd
import os

FILES_TO_FIX = [
    "../data/processed/labels.csv",
    "../data/frames/frame_data_all.csv"
]

def rename_column_sexual_to_family(files):
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        if "sexual_content" in df.columns:
            df.rename(columns={"sexual_content": "family_friendly"}, inplace=True)
            print(f"✅ Renamed 'sexual_content' to 'family_friendly' in: {file_path}")

        df.to_csv(file_path, index=False)

def correct_family_friendly_logic(files):
    """
    Converts mis-transformed values where original 'Yes' (sexual_content=True)
    became 'family_friendly' = Yes instead of No.
    We fix this inversion by flipping again: Yes<->No
    """
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        if "family_friendly" in df.columns:
            if df["family_friendly"].dtype == object:
                # Flip values
                df["family_friendly"] = df["family_friendly"].map({"Yes": "No", "No": "Yes"})
                df.to_csv(file_path, index=False)
                print(f"🔁 Flipped 'family_friendly' labels in: {file_path}")
            else:
                print(f"⚠️ 'family_friendly' column is not in string format: {file_path}")
        else:
            print(f"⚠️ No 'family_friendly' column in: {file_path}")


if __name__ == '__main__':
    correct_family_friendly_logic(FILES_TO_FIX)