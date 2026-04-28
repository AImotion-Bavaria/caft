import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNNSP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FeedForwardNNSP, self).__init__()

        self.fc1 = nn.Linear(num_features, 128) #128
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.5) #0.5

    def forward(self, x):

        output = F.relu(self.fc1(x))
        output = self.dropout(output)
        output = F.relu(self.fc2(output))
        output = self.dropout(output)
        output = F.relu(self.fc3(output))
        output = self.dropout(output)
        output = F.relu(self.fc4(output))
        output = self.fc5(output)

        return output

    
class FeedForwardNNRandomBlobs(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FeedForwardNNSP, self).__init__()

        self.fc1 = nn.Linear(num_features, 128) #128
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.5) #0.5

    def forward(self, x):

        output = F.relu(self.fc1(x))
        output = self.dropout(output)
        output = F.relu(self.fc2(output))
        output = self.dropout(output)
        output = F.relu(self.fc3(output))
        output = self.dropout(output)
        output = F.relu(self.fc4(output))
        output = self.fc5(output)

        return output
