# model.py - CNN Model for Vocaluity AI Vocal Detection

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import config
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from config import DEVICE, N_MELS


class VocaluityCNN(nn.Module):
    """
    Simple CNN for AI-generated music detection.
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPool
    - Global Average Pooling
    - Fully connected layers with dropout
    - Binary or multi-class output
    """
    
    def __init__(self, num_classes=2, input_channels=1):
        super(VocaluityCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Convolutional Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Global pooling to fixed size
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x
    
    def predict_proba(self, x):
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                proba = torch.softmax(logits, dim=1)
            else:
                proba = torch.softmax(logits, dim=1)
        return proba
    
    def predict(self, x):
        """Get class predictions."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


class VocaluityResNet(nn.Module):
    """
    ResNet-style model for more complex patterns.
    Uses residual connections for better gradient flow.
    """
    
    def __init__(self, num_classes=2, input_channels=1):
        super(VocaluityResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res1 = self._make_residual_block(64, 64)
        self.res2 = self._make_residual_block(64, 128, stride=2)
        self.res3 = self._make_residual_block(128, 256, stride=2)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResidualBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for quick prototyping.
    Faster training, suitable for initial experiments.
    """
    
    def __init__(self, num_classes=2, input_channels=1):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_type='simple', num_classes=2, input_channels=1):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: 'simple', 'resnet', or 'lightweight'
        num_classes: Number of output classes
        input_channels: Number of input channels
    
    Returns:
        Model instance
    """
    models = {
        'simple': VocaluityCNN,
        'resnet': VocaluityResNet,
        'lightweight': LightweightCNN
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type](num_classes=num_classes, input_channels=input_channels)
    model = model.to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


# Test the models
if __name__ == "__main__":
    print("Testing models...")
    
    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 1, 128, 216).to(DEVICE)
    
    print("\n" + "="*50)
    print("Testing Simple CNN")
    print("="*50)
    model = get_model('simple', num_classes=2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n" + "="*50)
    print("Testing ResNet")
    print("="*50)
    model = get_model('resnet', num_classes=6)  # 6 classes for multi-class
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n" + "="*50)
    print("Testing Lightweight CNN")
    print("="*50)
    model = get_model('lightweight', num_classes=2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
