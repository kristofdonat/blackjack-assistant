import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

#init the blackjack model
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
    
#load the trained model
net = BlackjackNet()
net.load_state_dict(torch.load("training/torch/blackjack_model.pth", weights_only=True))

#small test
input_data = torch.tensor([[3.0, 17.0]])
pred = net(input_data)
print(f"Draw another? {'Yes' if pred.item() > 0.5 else 'No'} ({pred.item():.2f})")