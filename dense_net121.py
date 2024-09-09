import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
    
    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x
        
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
        
class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_layers=[6, 12, 24, 16], num_classes=10):
        super(DenseNet121, self).__init__()
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Blocks and Transition Layers
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        in_channels = 64
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(in_channels, growth_rate, num_layers)
            self.dense_blocks.append(block)
            in_channels += num_layers * growth_rate
            if i < len(block_layers) - 1:
                transition = TransitionLayer(in_channels, in_channels // 2)
                self.transition_layers.append(transition)
                in_channels = in_channels // 2
        
        # Final Classification Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        for block, transition in zip(self.dense_blocks, self.transition_layers):
            x = block(x)
            x = transition(x)
        
        x = self.dense_blocks[-1](x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = DenseNet121(num_classes=10) # for cifar 10 classes
print(model)
