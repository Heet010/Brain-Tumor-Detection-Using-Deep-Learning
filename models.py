"""
Neural network models for brain tumor detection
Includes ResNet for classification and ResUNet for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional
import math


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
class BrainTumorResNet(nn.Module):
    """ResNet model for brain tumor classification"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(BrainTumorResNet, self).__init__()
        
        # Use pretrained ResNet50 as backbone
        if pretrained:
            self.backbone = models.resnet50(pretrained=True)
            # Freeze early layers for transfer learning
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False
        else:
            self.backbone = models.resnet50(pretrained=False)
        
        # Modify the classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class ResidualDoubleConv(nn.Module):
    """Residual double convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualDoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super(Down, self).__init__()
        if use_residual:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                ResidualDoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
            
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_residual: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_residual:
                self.conv = ResidualDoubleConv(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_residual:
                self.conv = ResidualDoubleConv(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels)
                
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class ResUNet(nn.Module):
    """ResUNet model for brain tumor segmentation"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = ResidualDoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, use_residual=True)
        self.down2 = Down(128, 256, use_residual=True)
        self.down3 = Down(256, 512, use_residual=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_residual=True)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, use_residual=True)
        self.up2 = Up(512, 256 // factor, bilinear, use_residual=True)
        self.up3 = Up(256, 128 // factor, bilinear, use_residual=True)
        self.up4 = Up(128, 64, bilinear, use_residual=True)
        self.outc = OutConv(64, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

class UNet(nn.Module):
    """Standard U-Net model for comparison"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, use_residual=False)
        self.down2 = Down(128, 256, use_residual=False)
        self.down3 = Down(256, 512, use_residual=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_residual=False)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, use_residual=False)
        self.up2 = Up(512, 256 // factor, bilinear, use_residual=False)
        self.up3 = Up(256, 128 // factor, bilinear, use_residual=False)
        self.up4 = Up(128, 64, bilinear, use_residual=False)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
    
class EnsembleModel(nn.Module):
    """Ensemble model combining classification and segmentation"""
    
    def __init__(self, classification_model, segmentation_model):
        super(EnsembleModel, self).__init__()
        self.classification_model = classification_model
        self.segmentation_model = segmentation_model
        
        # Freeze pretrained models initially
        for param in self.classification_model.parameters():
            param.requires_grad = False
        for param in self.segmentation_model.parameters():
            param.requires_grad = False
            
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 + 2, 64),  # 2 from classification + 2 from segmentation
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # Get classification predictions
        cls_out = self.classification_model(x)
        cls_prob = F.softmax(cls_out, dim=1)
        
        # Get segmentation predictions
        seg_out = self.segmentation_model(x)
        seg_prob = F.softmax(seg_out, dim=1)
        
        # Global average pooling for segmentation features
        seg_global = torch.mean(seg_prob, dim=(2, 3))
        
        # Combine features
        combined_features = torch.cat([cls_prob, seg_global], dim=1)
        
        # Final prediction
        final_out = self.fusion(combined_features)
        
        return {
            'classification': cls_out,
            'segmentation': seg_out,
            'ensemble': final_out
        }

class LightweightUNet(nn.Module):
    """Lightweight U-Net for low VRAM GPUs"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super(LightweightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder with much fewer channels
        self.enc1 = self._double_conv(n_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)   
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

def get_model(model_name: str, num_classes: int = 2, **kwargs):
    """Factory function to get different models"""
    
    if model_name.lower() == 'resnet':
        return BrainTumorResNet(num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'resunet':
        kwargs.pop('n_classes', None)
        return ResUNet(n_classes=num_classes, **kwargs)
    elif model_name.lower() == 'unet':
        kwargs.pop('n_classes', None)
        return UNet(n_classes=num_classes, **kwargs)
    elif model_name.lower() == 'lightweight_unet':
        kwargs.pop('n_classes', None)
        return LightweightUNet(n_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(1, 3, 256, 256)):
    """Print model summary"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                print("Output shapes:")
                for key, value in output.items():
                    print(f"  {key}: {value.shape}")
            else:
                print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test ResNet for classification
    print("\n=== ResNet Classification Model ===")
    resnet_model = get_model('resnet', num_classes=2, pretrained=True)
    resnet_model = resnet_model.to(device)
    model_summary(resnet_model)
    
    # Test ResUNet for segmentation
    print("\n=== ResUNet Segmentation Model ===")
    resunet_model = get_model('resunet', n_classes=2)
    resunet_model = resunet_model.to(device)
    model_summary(resunet_model)
    
    # Test UNet for segmentation
    print("\n=== UNet Segmentation Model ===")
    unet_model = get_model('unet', n_classes=2)
    unet_model = unet_model.to(device)
    model_summary(unet_model)
    
    # Test Ensemble Model
    print("\n=== Ensemble Model ===")
    ensemble_model = EnsembleModel(resnet_model, resunet_model)
    ensemble_model = ensemble_model.to(device)
    model_summary(ensemble_model)