import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to training log
LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "training_log.csv"
)


def plot_loss(log_path=LOG_PATH):
    if not os.path.exists(log_path):
        print(f"❌ Log file not found at: {log_path}")
        return

    df = pd.read_csv(log_path)

    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        print("❌ training_log.csv must contain 'epoch', 'train_loss', and 'val_loss'")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="s")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(log_path), "loss_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 👈 Ensure directory exists
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Loss curve saved to: {save_path}")


if __name__ == "__main__":
    plot_loss()

