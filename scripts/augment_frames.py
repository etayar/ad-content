import os
import uuid
import pandas as pd
from PIL import Image
from torchvision import transforms


def get_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
    ])


def augment_image(img, augmentations, n=3):
    return [augmentations(img) for _ in range(n)]


def save_augmented_images(augmented_imgs, original_row, save_dir):
    entries = []
    for img in augmented_imgs:
        new_id = str(uuid.uuid4())[:8]
        new_name = f"aug_{new_id}.jpg"
        new_path = os.path.join(save_dir, new_name)
        img.save(new_path)

        new_entry = original_row.copy()
        new_entry["frame_path"] = new_path
        new_entry["augmented"] = 1
        entries.append(new_entry)
    return entries


def augment_frames(input_csv, save_dir, output_csv, n_aug=3):
    os.makedirs(save_dir, exist_ok=True)
    frame_data = pd.read_csv(input_csv)
    augmented_entries = []

    augmentations = get_augmentations()

    for i, row in frame_data.iterrows():
        img_path = row["frame_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"Skipping unreadable image: {img_path}")
            continue

        aug_imgs = augment_image(img, augmentations, n=n_aug)
        augmented_entries.extend(save_augmented_images(aug_imgs, row.to_dict(), save_dir))

    frame_data["augmented"] = 0
    full_df = pd.concat([frame_data, pd.DataFrame(augmented_entries)], ignore_index=True)
    full_df.to_csv(output_csv, index=False)
    print(f"✅ Augmented dataset saved to {output_csv}")


if __name__ == "__main__":
    INPUT_CSV = "data/frames/frame_data_all.csv"
    SAVE_DIR = "data/frames/augmented_frames"
    META_CSV = os.path.join(SAVE_DIR, "augmented_metadata.csv")
    augment_frames(INPUT_CSV, SAVE_DIR, META_CSV, n_aug=3)
