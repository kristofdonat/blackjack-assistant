import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Config
# ----------------------------
DATA_PATH  = "training/torch/data/blkjckhands.csv"
MODEL_PATH = "blackjack_model.pth"

SEED = 42
BATCH_SIZE = 4096
EPOCHS = 6
LR = 1e-3
TRAIN_SPLIT = 0.9

CARD_COLS = ["card1", "card2", "card3", "card4", "card5"]

# ----------------------------
# Repro / device
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usable device: {device}")


# ----------------------------
# Blackjack total (Ace-aware), but we only USE the resulting total feature.
# This makes training totals match real "blackjack totals" the user will input.
# Works if Ace is stored as 11 or 1 in data.
# ----------------------------
def blackjack_total(cards):
    # normalize: treat aces as 1 first
    norm = [(1 if c in (1, 11) else int(c)) for c in cards]
    total = sum(norm)
    aces = sum(1 for c in cards if c in (1, 11))

    # upgrade some aces from 1 -> 11 (+10) if it doesn't bust
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1

    return total


def extract_cards_in_order(row):
    cards = []
    for c in CARD_COLS:
        v = int(getattr(row, c))
        if v != 0:
            cards.append(v)
    return cards


# ----------------------------
# Build decision-point dataset:
# For each state after 2+ cards:
#   X = [num_cards_now, total_now]
#   y = 1 if there exists a next card (player hit), else 0 (player stood)
# ----------------------------
def build_decision_dataset(df: pd.DataFrame):
    X_list, y_list = [], []

    for row in df.itertuples(index=False):
        cards = extract_cards_in_order(row)
        if len(cards) < 2:
            continue

        running = []
        for i, card in enumerate(cards):
            running.append(int(card))

            if i < 1:
                continue  # no decision until you have 2 cards

            total_now = blackjack_total(running)
            if total_now > 21:
                break  # busted states are terminal

            num_cards_now = i + 1
            hit_next = 1 if i < (len(cards) - 1) else 0

            X_list.append((num_cards_now, total_now))
            y_list.append((hit_next,))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


# ----------------------------
# Model (logits -> BCEWithLogitsLoss is more stable than Sigmoid+BCELoss)
# ----------------------------
class BlackjackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)


def accuracy_from_logits(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == y_true).float().mean().item()


def train_and_save():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    print("Loading CSV...")
    df = pd.read_csv(DATA_PATH)

    print("Building decision dataset...")
    X, y = build_decision_dataset(df)
    print(f"Decision samples: {len(X):,}")
    print(f"Hit rate (label mean): {float(y.mean()):.3f}")

    # shuffle + split
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * TRAIN_SPLIT)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # normalize features (important: save mean/std for inference)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = BlackjackNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_acc += accuracy_from_logits(logits.detach(), yb)
            n += 1

        tr_loss /= max(n, 1)
        tr_acc /= max(n, 1)

        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        m = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                va_loss += loss.item()
                va_acc += accuracy_from_logits(logits, yb)
                m += 1

        va_loss /= max(m, 1)
        va_acc /= max(m, 1)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    # save model + scaler (so inference matches training)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": ["num_cards", "total"],
    }
    torch.save(ckpt, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


def load_model_for_inference(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=device)
    model = BlackjackNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]
    return model, mean, std


@torch.no_grad()
def predict_hit(model, mean, std, num_cards: int, total: int, threshold: float = 0.5):
    x = np.array([[float(num_cards), float(total)]], dtype=np.float32)
    x = (x - mean) / std
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    logits = model(xt)
    prob = torch.sigmoid(logits).item()
    return (prob >= threshold), prob


if __name__ == "__main__":
    train_and_save()

    model, mean, std = load_model_for_inference()

    # Example: 3 cards totaling 17
    hit, prob = predict_hit(model, mean, std, num_cards=3, total=17)
    print(f"(3,17) -> HIT? {'Yes' if hit else 'No'}  (p={prob:.3f})")

    # Your old test
    hit, prob = predict_hit(model, mean, std, num_cards=2, total=14)
    print(f"(2,14) -> HIT? {'Yes' if hit else 'No'}  (p={prob:.3f})")