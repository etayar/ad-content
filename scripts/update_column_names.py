import pandas as pd
import os

FILES = [
    "data/processed/labels.csv",
    "data/frames/frame_data_all.csv"
]

for file_path in FILES:
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if "sexual_content" in df.columns:
        df.rename(columns={"sexual_content": "family_friendly"}, inplace=True)

        if df["family_friendly"].dtype == object:
            df["family_friendly"] = df["family_friendly"].map({"Yes": False, "No": True})

        df.to_csv(file_path, index=False)
        print(f"✅ Updated and saved: {file_path}")
    else:
        print(f"⚠️ No 'sexual_content' column in: {file_path}")
