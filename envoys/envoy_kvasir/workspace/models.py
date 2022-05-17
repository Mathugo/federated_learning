"""
DeepLab model definition 
"""
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, deeplabv3_resnet101
from loss import DoubleConv, Down, Up
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

class DeepLabv3:
    def build_deeplab(self, num_classes, num_features_fc: int=256, backbone: str="mobilenetv3", pretrained_backbone: bool=True, pretrained_head: bool=True, alpha: float=0):
        """ change the output layer and add a freezing coeff.
            the number of in channel for the DeepLabHead depends on the backbone, 
            MobileNetv3 has 960 out channel whereas resnet101 2048
        Args: 
            backbone: str='mobilenetv3' "pre-trained backbone to download"
        """
        if backbone == "mobilenetv3":
            if pretrained_head:
                self.model = deeplabv3_mobilenet_v3_large(pretrained=True, pretrained_backbone=True)
                out_channel = 960
                self.model.classifier = DeepLabHead(out_channel, num_classes)
                self.model.aux_classifier = nn.Identity()
                print("[*] Changing head for {} classes and removing aux classifier".format(num_classes))
            else:
                self.model = deeplabv3_mobilenet_v3_large(pretrained_backbone=True, num_classes=num_classes)
        elif backbone == "resnet101":
            if pretrained_head:
                self.model= deeplabv3_resnet101(pretrained=True, pretrained_backbone=True)
                out_channel = 2048
                self.model.classifier = DeepLabHead(out_channel, num_classes)
                self.model.aux_classifier = nn.Identity()
                print("[*] Changing head for {} classes and removing aux classifier".format(num_classes))
            else:
                self.model= deeplabv3_resnet101(pretrained_backbone=True, num_classes=num_classes)
        else:
            assert "No such backbone"

        if alpha == 0:
            self.dofreeze = False
        else:
            self.dofreeze = True
            self.alpha=alpha
            print("[!] This model will be trained using alpha freezing coef = {} meaning {}/{} layers will be freeze".format(self.alpha, int(self.alpha*sum(1 for x in self.model.parameters())), sum(1 for x in self.model.parameters())))
        return self.model
 
    def freeze(self):
        if self.dofreeze:
            s = sum(1 for x in self.model.parameters())
            l_freeze = int(s*self.alpha)
            print("{} layers in this model, freezing {} layer\n".format(s, l_freeze))
            for i,param in enumerate(self.model.parameters()):
                param.requires_grad = False
                if l_freeze < i:
                    break
            for name, layer in self.model.named_modules():
                print(name, layer)
    
    def unfreeze(self):
        for i,param in enumerate(self.model.parameters()):
            param.requires_grad = True
