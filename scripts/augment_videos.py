import os
import cv2
import uuid
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path


def get_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
    ])


def extract_frames_from_video(video_path, interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def save_image(image_array, save_dir, prefix="frame"):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    new_id = str(uuid.uuid4())[:8]
    new_name = f"{prefix}_{new_id}.jpg"
    new_path = os.path.join(save_dir, new_name)
    img.save(new_path)
    return new_path


def save_augmented_images(img_pil, save_dir, base_meta, augmentations, n_aug=3):
    entries = []
    for _ in range(n_aug):
        aug_img = augmentations(img_pil)
        new_id = str(uuid.uuid4())[:8]
        new_name = f"aug_{new_id}.jpg"
        new_path = os.path.join(save_dir, new_name)
        aug_img.save(new_path)

        aug_meta = base_meta.copy()
        aug_meta["frame_path"] = new_path
        aug_meta["augmented"] = 1
        entries.append(aug_meta)
    return entries


def process_videos(video_dir, save_dir, output_csv, interval_sec=1, n_aug=3):
    os.makedirs(save_dir, exist_ok=True)
    video_paths = list(Path(video_dir).glob("*.mp4"))

    all_entries = []
    augmentations = get_augmentations()

    for video_path in video_paths:
        frames = extract_frames_from_video(str(video_path), interval_sec=interval_sec)
        for frame in frames:
            frame_path = save_image(frame, save_dir, prefix="orig")
            base_meta = {
                "video_name": video_path.name,
                "frame_path": frame_path,
                "augmented": 0
            }

            # Augment the original frame
            img_pil = Image.open(frame_path).convert("RGB")
            aug_entries = save_augmented_images(img_pil, save_dir, base_meta, augmentations, n_aug=n_aug)

            all_entries.append(base_meta)
            all_entries.extend(aug_entries)

    pd.DataFrame(all_entries).to_csv(output_csv, index=False)
    print(f"✅ Video frames and augmentations saved to {output_csv}")


if __name__ == "__main__":
    VIDEO_DIR = "data/videos"
    SAVE_DIR = "data/frames/from_videos"
    OUTPUT_CSV = os.path.join(SAVE_DIR, "video_augmented_metadata.csv")
    process_videos(VIDEO_DIR, SAVE_DIR, OUTPUT_CSV, interval_sec=1, n_aug=3)
