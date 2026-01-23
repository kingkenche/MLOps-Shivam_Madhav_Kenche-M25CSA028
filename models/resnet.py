"""
ResNet implementations for MNIST and FashionMNIST
Models: ResNet-18, ResNet-32, ResNet-50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/32"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture adapted for MNIST/FashionMNIST (28x28 images)"""
    
    def __init__(self, block, num_blocks, num_classes=10, in_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution - adapted for 28x28 images
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10, in_channels=1, pretrained=False):
    """ResNet-18 for MNIST/FashionMNIST"""
    if pretrained:
        raise ValueError("Pretrained weights not supported for this assignment")
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def ResNet32(num_classes=10, in_channels=1, pretrained=False):
    """ResNet-32 for MNIST/FashionMNIST"""
    if pretrained:
        raise ValueError("Pretrained weights not supported for this assignment")
    return ResNet(BasicBlock, [5, 5, 5, 5], num_classes=num_classes, in_channels=in_channels)


def ResNet50(num_classes=10, in_channels=1, pretrained=False):
    """ResNet-50 for MNIST/FashionMNIST"""
    if pretrained:
        raise ValueError("Pretrained weights not supported for this assignment")
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


# Test the models
if __name__ == "__main__":
    # Test ResNet-18
    model18 = ResNet18()
    x = torch.randn(1, 1, 28, 28)
    y = model18(x)
    print(f"ResNet-18 output shape: {y.shape}")
    print(f"ResNet-18 parameters: {sum(p.numel() for p in model18.parameters()):,}")
    
    # Test ResNet-32
    model32 = ResNet32()
    y = model32(x)
    print(f"ResNet-32 output shape: {y.shape}")
    print(f"ResNet-32 parameters: {sum(p.numel() for p in model32.parameters()):,}")
    
    # Test ResNet-50
    model50 = ResNet50()
    y = model50(x)
    print(f"ResNet-50 output shape: {y.shape}")
    print(f"ResNet-50 parameters: {sum(p.numel() for p in model50.parameters()):,}")
