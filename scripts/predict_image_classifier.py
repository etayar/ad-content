import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import joblib
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


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            labels = targets["family_friendly"].float().to(device)
            _, _, out_family = model(images)
            pred_probs = torch.sigmoid(out_family)
            all_pred.extend(pred_probs.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_true, all_pred)
    print(f"ROC-AUC (family_friendly): {auc:.4f}")


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

    dataset, df, _ = load_and_prepare_dataset("data/frames/frame_data_all.csv", transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model_cls = MultiLabelResNet if args.model == "resnet" else MultiLabelViT
    model = model_cls(
        df["industry_encoded"].nunique(),
        df["audience_encoded"].nunique()
    ).to(device)

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded model weights from {args.weights}")
    else:
        print("⚠️ No weights provided. Evaluating with untrained model.")

    evaluate(model, loader, device)


if __name__ == "__main__":
    # python scripts/predict_image_classifier.py --image path/to/frame.jpg --model resnet --weights models/multilabel_resnet.pt
    main()
