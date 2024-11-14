import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.conv1(x)       # Input(3, 32, 32), Output(16, 28, 28)
        x = self.relu(x)
        x = self.pool1(x)       # Output(16, 14, 14)
        x = self.conv2(x)       # Output(32, 10, 10)
        x = self.relu(x)
        x = self.pool2(x)       # Output(32, 5, 5)
        x = self.flatten(x)     # Output(32*5*5)
        x = self.fc1(x)         # Output(120)
        x = self.relu(x)
        x = self.fc2(x)         # Output(84)
        x = self.relu(x)
        out = self.fc3(x)       # Output(10)

        return out
