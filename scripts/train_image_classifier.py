import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import argparse

# === Argument parser ===
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
args = parser.parse_args()

# === Load and preprocess metadata ===
data_path = "data/frames/frame_data_all.csv"
df = pd.read_csv(data_path)

# Encode labels
label_cols = ["industry", "audience", "family_friendly"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le

    # Save encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, f"models/label_encoder_{col}.pkl")

# === Define Dataset ===
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

# === Define Model ===
class MultiLabelResNet(nn.Module):
    def __init__(self, num_industries, num_audiences):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_industry = nn.Linear(in_features, num_industries)
        self.fc_audience = nn.Linear(in_features, num_audiences)
        self.fc_family = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        out_industry = self.fc_industry(features)
        out_audience = self.fc_audience(features)
        out_family = self.fc_family(features)
        return out_industry, out_audience, out_family.squeeze(1)

# === Prepare ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset
full_dataset = FrameDataset(df, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = MultiLabelResNet(
    num_industries=df["industry_encoded"].nunique(),
    num_audiences=df["audience_encoded"].nunique()
).to(device)

criterion_industry = nn.CrossEntropyLoss()
criterion_audience = nn.CrossEntropyLoss()
criterion_family = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Track Losses ===
train_losses = []
val_losses = []

# === Training Loop ===
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = images.to(device)
        industry_labels = targets["industry"].to(device)
        audience_labels = targets["audience"].to(device)
        family_labels = targets["family_friendly"].float().to(device)

        optimizer.zero_grad()
        out_industry, out_audience, out_family = model(images)

        loss_industry = criterion_industry(out_industry, industry_labels)
        loss_audience = criterion_audience(out_audience, audience_labels)
        loss_family = criterion_family(out_family, family_labels)

        loss = loss_industry + loss_audience + loss_family
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # === Validation Loss ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            industry_labels = targets["industry"].to(device)
            audience_labels = targets["audience"].to(device)
            family_labels = targets["family_friendly"].float().to(device)

            out_industry, out_audience, out_family = model(images)

            loss_industry = criterion_industry(out_industry, industry_labels)
            loss_audience = criterion_audience(out_audience, audience_labels)
            loss_family = criterion_family(out_family, family_labels)

            val_loss += (loss_industry + loss_audience + loss_family).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"\n[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Save training log
log_df = pd.DataFrame({"epoch": list(range(1, args.epochs + 1)), "train_loss": train_losses, "val_loss": val_losses})
os.makedirs("logs", exist_ok=True)
log_df.to_csv("logs/training_log.csv", index=False)
print("\n📝 Training log saved to logs/training_log.csv")

# === Validation ===
model.eval()
all_true_family, all_pred_family = [], []
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        family_labels = targets["family_friendly"].float().to(device)
        _, _, out_family = model(images)

        preds = torch.sigmoid(out_family).cpu().numpy()
        truths = family_labels.cpu().numpy()

        all_pred_family.extend(preds)
        all_true_family.extend(truths)

# Compute ROC-AUC
try:
    auc_score = roc_auc_score(all_true_family, all_pred_family)
    print(f"\nValidation ROC-AUC (family_friendly): {auc_score:.4f}")
except ValueError:
    print("Not enough data for ROC-AUC computation.")

# === Save Model ===
torch.save(model.state_dict(), "models/multilabel_resnet50.pt")
print("\n✅ Model saved to models/multilabel_resnet50.pt")
