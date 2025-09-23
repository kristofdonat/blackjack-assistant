import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usable device: {device}")


#Load data
df = pd.read_csv("training/torch/data/blkjckhands.csv")

print(f"Data loaded with {len(df)} rows")

#Listing of card columns
card_cols = ["card1", "card2", "card3", "card4", "card5"]

def get_x1(cards):
    return sum(1 for c in cards if c != 0)

def get_x2(cards, total):
    nonzero = [c for c in cards if c != 0]
    return total - nonzero[-1] if len(nonzero) > 2 else total

def get_y(cards):
    return 1 if sum(1 for c in cards if c != 0) > 2 else 0

x, y = [], []
for _, row in df.iterrows():
    cards = [row[c] for c in card_cols]
    x1 = get_x1(cards)
    x2 = get_x2(cards, row["sumofcards"])
    label = get_y(cards)
    x.append([x1, x2])
    y.append(label)
    print(f"Processed hand: cards={cards}, x1={x1}, x2={x2}, y={label}")

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


_ = input(f"Data preload finished with {len(x)} items, waiting press any key to load")

print("pressed key, loading..")


class BlackjackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  #Binaric output
        )

    def forward(self, x):
        return self.model(x)

net = BlackjackNet()

criterion = nn.BCELoss()  #Binary Cross Entropy Loss
optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 5000
for epoch in range(epochs):
    outputs = net(x)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

torch.save(net.state_dict(), "training/torch/blackjack_model.pth")


input_data = torch.tensor([[2.0, 14.0]])
pred = net(input_data)
print(f"Draw another? {'Yes' if pred.item() > 0.5 else 'No'} ({pred.item():.2f})")
