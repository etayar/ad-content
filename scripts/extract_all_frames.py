import os
import pandas as pd
from investigate_video_audio import extract_video_and_audio_data


def batch_extract_all_videos(
    label_csv_path="data/processed/labels.csv",
    raw_video_dir="data/raw",
    output_dir="data/frames/extracted_frames",
    frame_interval=5
):
    """
    Extracts frames and audio from all labeled videos.

    Parameters:
    - label_csv_path: path to CSV containing `video_name` column
    - raw_video_dir: directory where raw videos are stored
    - output_dir: where to store extracted frames/audio
    - frame_interval: extract every Nth frame (e.g., 5)
    """
    df = pd.read_csv(label_csv_path)

    for video_name in df["video_name"]:
        video_path = os.path.join(raw_video_dir, video_name)

        if not os.path.exists(video_path):
            print(f"❌ Skipping missing file: {video_path}")
            continue

        print(f"\n🎞 Processing: {video_name}")
        try:
            extract_video_and_audio_data(
                video_path=video_path,
                output_dir=output_dir,
                save=True,
                frame_interval=frame_interval
            )
        except Exception as e:
            print(f"⚠️ Failed to process {video_name}: {e}")


if __name__ == "__main__":
    batch_extract_all_videos()
