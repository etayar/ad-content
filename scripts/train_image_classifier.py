import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_auc_score
import joblib
import argparse
from tqdm import tqdm
from datetime import datetime
import sys


# Absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


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
        img_path = resolve_path(row["frame_path"])
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


def train_model(model, train_loader, val_loader, device, model_filename, epochs=10, fine_tune_backbone=True):
    criterion_industry = nn.CrossEntropyLoss()
    criterion_audience = nn.CrossEntropyLoss()
    criterion_family = nn.BCEWithLogitsLoss()

    if not fine_tune_backbone:
        print("Freezing backbone weights...")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning entire model (backbone + heads)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    train_losses, val_losses = [], []
    train_batch_losses, val_batch_losses = [], []
    best_auc = 0.0

    industry_aucs, audience_aucs, family_aucs, avg_aucs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for images, targets in tqdm(train_loader, desc="Training", leave=False):
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

            train_batch_losses.append({
                "epoch": epoch + 1,
                "batch_loss": loss.item(),
                "phase": "train"
            })

            if (i := len(train_batch_losses)) % 10 == 0:
                tqdm.write(f"[Train][Batch {i}] Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0

        true_industry, true_audience, true_family = [], [], []
        pred_industry, pred_audience, pred_family = [], [], []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", leave=False):
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

                val_batch_losses.append({
                    "epoch": epoch + 1,
                    "batch_loss": loss.item(),
                    "phase": "val"
                })

                if len(val_batch_losses) % 5 == 0:
                    tqdm.write(f"[Val][Batch {len(val_batch_losses)}] Loss: {loss.item():.4f}")

                true_industry.extend(industry_labels.cpu().numpy())
                true_audience.extend(audience_labels.cpu().numpy())
                true_family.extend(family_labels.cpu().numpy())

                pred_industry.extend(torch.softmax(out_industry, dim=1).cpu().numpy())
                pred_audience.extend(torch.softmax(out_audience, dim=1).cpu().numpy())
                pred_family.extend(torch.sigmoid(out_family).cpu().numpy())

        num_industries = model.fc_industry.out_features
        num_audiences = model.fc_audience.out_features

        try:
            industry_auc = roc_auc_score(
                label_binarize(true_industry, classes=list(range(num_industries))),
                pred_industry,
                average="macro",
                multi_class="ovr"
            )
        except:
            industry_auc = 0.0

        try:
            audience_auc = roc_auc_score(
                label_binarize(true_audience, classes=list(range(num_audiences))),
                pred_audience,
                average="macro",
                multi_class="ovr"
            )
        except:
            audience_auc = 0.0

        try:
            family_auc = roc_auc_score(true_family, pred_family)
        except:
            family_auc = 0.0

        epoch_auc = (industry_auc + audience_auc + family_auc) / 3

        industry_aucs.append(industry_auc)
        audience_aucs.append(audience_auc)
        family_aucs.append(family_auc)
        avg_aucs.append(epoch_auc)

        print(f"\nEpoch {epoch + 1} ROC-AUCs — Industry: {industry_auc:.4f}, Audience: {audience_auc:.4f}, Family: {family_auc:.4f}")
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train:.4f}, Validation Loss: {avg_val:.4f}")

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_model_path = os.path.join("models", f"best_{model_filename}")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated (Avg AUC {epoch_auc:.4f}) → {best_model_path}")

    os.makedirs("scripts/logs", exist_ok=True)

    pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "model_file": model_filename
    }).to_csv("scripts/logs/training_log.csv", index=False)

    pd.DataFrame(train_batch_losses + val_batch_losses).to_csv("scripts/logs/batch_loss_log.csv", index=False)

    pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "industry_auc": industry_aucs,
        "audience_auc": audience_aucs,
        "family_auc": family_aucs,
        "avg_auc": avg_aucs
    }).to_csv("scripts/logs/auc_log.csv", index=False)

    print("Logs saved to scripts/logs/")
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


def main(config_d=None):
    if config_d:
        config_args = []
        for k, v in config_d.items():
            if isinstance(v, bool) and v:  # only include flag if it's True
                config_args.append(f"--{k}")
            elif not isinstance(v, bool):
                config_args.append(f"--{k}={v}")
        args = parse_args(config_args)
    else:
        args = parse_args()

    data_path = resolve_path(args.data_path)
    df, _ = load_data(data_path=data_path)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"multilabel_{args.model}_{timestamp}.pt"

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        fine_tune_backbone=args.fine_tune_backbone,
        model_filename=model_filename
    )

    evaluate_family_friendly(model, val_loader, device)

    model_path = os.path.join("models", model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vit"], help="Model architecture")
    parser.add_argument("--data_path", type=str, default="data/frames/frame_data_all.csv", help="Path to data CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--fine_tune_backbone", action="store_true", help="Whether to fine-tune the backbone")
    return parser.parse_args(args)


if __name__ == "__main__":
    # python scripts/train_image_classifier.py \
    #   --epochs 20 \
    #   --batch_size 64 \
    #   --model vit \
    #   --data_path data/frames/frame_data_all.csv \
    #   --fine_tune_backbone
    #
    # If you include --fine_tune_backbone, the backbone will be trained.
    #
    # If you omit it, only the custom FC heads will be trained (backbone remains frozen).
    if len(sys.argv) > 1:
        main()  # CLI args are provided
    else:
        config_d = {
            "epochs": 1,
            "model": "resnet",
            "data_path": "data/frames/frame_data_all.csv",
            "batch_size": 64,
            "fine_tune_backbone": True
        }
        main(config_d)
