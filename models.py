"""
Defines custom torch.nn.Module classes
"""

import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(176 * 144, 64)
        self.layer2 = nn.Linear(64, 4)

    def forward(self, img):
        flattened = img.view(-1, 176 * 144)
        activation1 = F.relu(self.layer1(flattened))
        output = self.layer2(activation1)
        return output

class CNN_Basic(nn.Module):
    def __init__(self):
        super(CNN_Basic, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10 * 19 * 15, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 19 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 7, 1, 6) # 182x150 -> pool 91x75
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5, 1, 4) # 95x79 -> pool 47x39
        self.conv3 = nn.Conv2d(10, 20, 3, 1, 2) # 49x41 -> pool 24x20
        self.fc1 = nn.Linear(20 * 24 * 20, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 24 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

####################################################################################
# nn.Conv2d(in_channels, out_channels, kernel_size[, stride, padding])
# nn.MaxPool2d(kernel_size, stride[, padding])
# nn.Linear(in_features, out_features)

# n_out = [(n_in + 2p - k) / s] + 1
#   n_out = output size, n_in = input size, p = padding, k = kernel size, s = stride
####################################################################################
