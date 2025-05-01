import os
import pandas as pd
import torch
import torch.nn as nn
from networkx import config
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vit"], help="Model architecture")
    parser.add_argument("--data_path", type=str, default="data/frames/frame_data_all.csv", help="path to data")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_data(data_path):
    df = pd.read_csv(data_path)

    label_cols = ["industry", "audience", "family_friendly"]
    label_encoders = {}
    os.makedirs("models", exist_ok=True)

    for col in label_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
        joblib.dump(le, f"models/label_encoder_{col}.pkl")

    return df, label_encoders


class FrameDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["frame_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = {
            "industry": row["industry_encoded"],
            "audience": row["audience_encoded"],
            "family_friendly": row["family_friendly_encoded"]
        }
        return image, labels


class MultiLabelResNet(nn.Module):
    def __init__(self, num_industries, num_audiences, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_industry = nn.Linear(in_features, num_industries)
        self.fc_audience = nn.Linear(in_features, num_audiences)
        self.fc_family = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.fc_industry(features),
            self.fc_audience(features),
            self.fc_family(features).squeeze(1)
        )


class MultiLabelViT(nn.Module):
    def __init__(self, num_industries, num_audiences, pretrained=True):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.fc_industry = nn.Linear(in_features, num_industries)
        self.fc_audience = nn.Linear(in_features, num_audiences)
        self.fc_family = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.fc_industry(features),
            self.fc_audience(features),
            self.fc_family(features).squeeze(1)
        )


def train_model(model, train_loader, val_loader, device, epochs):
    criterion_industry = nn.CrossEntropyLoss()
    criterion_audience = nn.CrossEntropyLoss()
    criterion_family = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            industry_labels = targets["industry"].to(device)
            audience_labels = targets["audience"].to(device)
            family_labels = targets["family_friendly"].float().to(device)

            optimizer.zero_grad()
            out_industry, out_audience, out_family = model(images)
            loss = (
                criterion_industry(out_industry, industry_labels)
                + criterion_audience(out_audience, audience_labels)
                + criterion_family(out_family, family_labels)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                industry_labels = targets["industry"].to(device)
                audience_labels = targets["audience"].to(device)
                family_labels = targets["family_friendly"].float().to(device)

                out_industry, out_audience, out_family = model(images)
                loss = (
                    criterion_industry(out_industry, industry_labels)
                    + criterion_audience(out_audience, audience_labels)
                    + criterion_family(out_family, family_labels)
                )
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"\n[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train:.4f}, Validation Loss: {avg_val:.4f}")

    return train_losses, val_losses


def evaluate_family_friendly(model, val_loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            family_labels = targets["family_friendly"].float().to(device)
            _, _, out_family = model(images)
            all_pred.extend(torch.sigmoid(out_family).cpu().numpy())
            all_true.extend(family_labels.cpu().numpy())

    try:
        auc = roc_auc_score(all_true, all_pred)
        print(f"\nValidation ROC-AUC (family_friendly): {auc:.4f}")
    except ValueError:
        print("Not enough data for ROC-AUC computation.")


def main():
    args = parse_args()
    df, _ = load_data(data_path=args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FrameDataset(df, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_cls = MultiLabelResNet if args.model == "resnet" else MultiLabelViT
    model = model_cls(
        df["industry_encoded"].nunique(),
        df["audience_encoded"].nunique(),
        pretrained=True
    ).to(device)

    train_losses, val_losses = train_model(model, train_loader, val_loader, device, args.epochs)

    os.makedirs("logs", exist_ok=True)
    pd.DataFrame({"epoch": range(1, args.epochs + 1), "train_loss": train_losses, "val_loss": val_losses})\
        .to_csv("logs/training_log.csv", index=False)
    print("\n📝 Training log saved to logs/training_log.csv")

    evaluate_family_friendly(model, val_loader, device)
    torch.save(model.state_dict(), f"models/multilabel_{args.model}.pt")
    print(f"\n✅ Model saved to models/multilabel_{args.model}.pt")


if __name__ == "__main__":
    config = {}
    main()
