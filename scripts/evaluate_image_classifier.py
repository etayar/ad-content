import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import joblib
from collections import defaultdict
from train_image_classifier import FrameDataset, MultiLabelResNet, MultiLabelViT
from sklearn.metrics import accuracy_score, roc_auc_score

def load_and_prepare_dataset(data_path, transform):
    df = pd.read_csv(data_path)
    label_encoders = {}
    for col in ["industry", "audience", "family_friendly"]:
        le_path = f"models/label_encoder_{col}.pkl"
        label_encoders[col] = joblib.load(le_path)
        df[col + "_encoded"] = label_encoders[col].transform(df[col])
    dataset = FrameDataset(df, transform)
    return dataset, df, label_encoders

def evaluate(model, loader, device, df_frames, label_encoders):
    model.eval()
    frame_predictions = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            video_names = targets["video_name"]
            _, _, out_family = model(images)
            preds = torch.sigmoid(out_family).cpu().numpy()

            for name, pred in zip(video_names, preds):
                frame_predictions.append((name, float(pred)))

    # --- Aggregate by video ---
    video_preds = defaultdict(list)
    for name, pred in frame_predictions:
        video_preds[name].append(pred)

    results = {}
    for video, preds in video_preds.items():
        any_negative = any(p < 0.5 for p in preds)
        results[video] = "No" if any_negative else "Yes"

    print("\n📊 Video-Level Family-Friendly Classification")
    for video, prediction in results.items():
        print(f"{video}: {prediction}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "vit"], default="resnet")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset, df, label_encoders = load_and_prepare_dataset("data/frames/frame_data_all.csv", transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model_cls = MultiLabelResNet if args.model == "resnet" else MultiLabelViT
    model = model_cls(
        df["industry_encoded"].nunique(),
        df["audience_encoded"].nunique(),
        pretrained=True
    ).to(device)

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded model weights from {args.weights}")
    else:
        print("⚠️ No weights provided. Evaluating with untrained model.")

    evaluate(model, loader, device, df, label_encoders)

if __name__ == "__main__":
    # For ResNet-based model
    # python scripts/evaluate_video_classifier.py --model resnet --weights models/multilabel_resnet.pt
    #
    # For ViT-based model
    # python scripts/evaluate_video_classifier.py --model vit --weights models/multilabel_vit.pt
    main()
