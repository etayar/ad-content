import os
import pandas as pd

LABEL_PATH = "../data/processed/labels.csv"
EXTRACTED_BASE = "../data/frames/extracted_frames"
OUTPUT_PATH = "../data/frames/frame_data_all.csv"

# Load video-level labels
df_labels = pd.read_csv(LABEL_PATH)
label_dict = df_labels.set_index("video_name").to_dict(orient="index")

# Collect all per-video metadata
all_metadata = []

for video_folder in os.listdir(EXTRACTED_BASE):
    video_dir = os.path.join(EXTRACTED_BASE, video_folder)
    metadata_csv = os.path.join(video_dir, "metadata.csv")

    if not os.path.isfile(metadata_csv):
        print(f"⚠️ Skipping (no metadata): {metadata_csv}")
        continue

    frame_df = pd.read_csv(metadata_csv)

    # Add labels to each row
    labels = label_dict.get(video_folder + ".mp4") or label_dict.get(video_folder + ".mov")
    if labels is None:
        print(f"❌ No labels found for {video_folder}, skipping.")
        continue

    for key in ["industry", "audience", "family_friendly"]:
        frame_df[key] = labels[key]

    all_metadata.append(frame_df)

# Concatenate and save
full_df = pd.concat(all_metadata, ignore_index=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
full_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Aggregated frame metadata saved to: {OUTPUT_PATH}")
