import os
import pandas as pd
import matplotlib.pyplot as plt

# Base log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOSS_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
AUC_LOG_PATH = os.path.join(LOG_DIR, "auc_log.csv")


def plot_combined(loss_path=LOSS_LOG_PATH, auc_path=AUC_LOG_PATH):
    if not os.path.exists(loss_path):
        print(f"❌ Loss log not found at: {loss_path}")
        return
    if not os.path.exists(auc_path):
        print(f"❌ AUC log not found at: {auc_path}")
        return

    df_loss = pd.read_csv(loss_path)
    df_auc = pd.read_csv(auc_path)

    if not {"epoch", "train_loss", "val_loss"}.issubset(df_loss.columns):
        print("❌ training_log.csv must contain 'epoch', 'train_loss', and 'val_loss'")
        return

    required_auc_cols = {"epoch", "industry_auc", "audience_auc", "family_auc", "avg_auc"}
    if not required_auc_cols.issubset(df_auc.columns):
        print(f"❌ auc_log.csv must contain: {', '.join(required_auc_cols)}")
        return

    # Create combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot loss
    ax1.plot(df_loss["epoch"], df_loss["train_loss"], label="Train Loss", marker="o")
    ax1.plot(df_loss["epoch"], df_loss["val_loss"], label="Validation Loss", marker="s")
    ax1.set_title("Training vs. Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # Plot AUCs
    ax2.plot(df_auc["epoch"], df_auc["industry_auc"], label="Industry AUC", marker="o")
    ax2.plot(df_auc["epoch"], df_auc["audience_auc"], label="Audience AUC", marker="s")
    ax2.plot(df_auc["epoch"], df_auc["family_auc"], label="Family AUC", marker="^")
    ax2.plot(df_auc["epoch"], df_auc["avg_auc"], label="Average AUC", linestyle="--", linewidth=2)
    ax2.set_title("ROC-AUC per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "combined_metrics_plot.png")
    plt.savefig(save_path)
    plt.show(block=True)
    print(f"Combined loss + AUC plot saved to: {save_path}")


if __name__ == "__main__":
    plot_combined()
