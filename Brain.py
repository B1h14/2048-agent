import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10,4)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        x = torch.sigmoid(nn.Sigmoid()(self.fc1(x)))
        x = torch.sigmoid(nn.Sigmoid()(self.fc2(x)))
        x = torch.sigmoid(nn.Sigmoid ()(self.fc3(x)))
        x = self.softmax(x)
        return x

