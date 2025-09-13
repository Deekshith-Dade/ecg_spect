import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGClassifier(nn.Module):
    
    def __init__(self, in_channels:int, num_classes: int):
        super(ECGClassifier, self).__init__()
        
        # The input spectrogram has 8 channels (one for each ECG lead)
        # Input shape: (batch_size, 8, 129, 40)

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape after Pool1: (batch_size, 32, 256, 256)


        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape after Pool2: (batch_size, 32, 32, 10)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape after Pool3: (batch_size, 64, 16, 5)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = 256 * 16 * 16

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input batch of spectrograms.
        
        Returns:
            torch.Tensor: The model's output logits.
        """
        # Pass through convolutional blocks
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.flatten_size)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x