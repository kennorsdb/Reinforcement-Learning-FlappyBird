import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#TODO: Cambiar el size
INPUT_SIZE = 7344

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 12, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=11, stride=1)
        self.conv2_drop = nn.Dropout2d()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(INPUT_SIZE, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # Max pooling over a (3, 3) window
        x.unsqueeze_(0)
        cl1 = F.max_pool2d(F.relu(self.conv1(x)),3)
        # If the size is a square you can only specify a single number
        cl2 = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(cl1))), 2)
        cl3 = F.max_pool2d(F.relu(self.conv3(cl2)), 2)

        #TODO: ver como cambiarle esto
        v = cl3.view(-1, INPUT_SIZE)

        out = F.softmax(self.fc1(v), dim=1)
        return out

        