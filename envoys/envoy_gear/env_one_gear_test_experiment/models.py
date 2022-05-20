"""
DeepLab model definition 
"""
from typing import List
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
    def build_deeplab(self, num_classes=1, backbone: str="mobilenetv3", pretrained_backbone: bool=True, pretrained_head: bool=True, keep_features: bool=True, alpha: float=0, load_from_pkl: bool=False):
        """ Build a custom deeplabv3 model for semantic segmentation 

        Args: 
            pretrained_head: bool=True          " Enable pretrained deeplabhead on COCO2017 dataset"
            backbone: str='mobilenetv3'         " Pretrained backbone to download - mobilenetv3 or resnet101"
            pretrained_backbone: bool=True      " Enable pretrained backbone 
            load_from_pkl: bool=False           " Load the best model in the save folder"
            alpha: float=0                      " Freezing coeff -> 0 <= alpha <= 1 if alpha=1 all layers are freezed
        """
        self.alpha=alpha
        if load_from_pkl:
            print("[*] Loading best model ..")
            self.model = torch.load("save/best_model.pkl")
            print("[*] Done")
        else:
            print("[*] Building model ..")                
            if backbone == "mobilenetv3":
                if  pretrained_head:
                    self.model = deeplabv3_mobilenet_v3_large(pretrained=True, pretrained_backbone=pretrained_backbone)
                    out_channel = 960
                    if keep_features:
                        print("[*] Keeping features of deeplabv3, freezing all the model except the head")
                        self.dofreeze = True
                        self.freeze(1)
                    #self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))
                    self.model.classifier = DeepLabHead(out_channel, num_classes)
                    self.model.aux_classifier = nn.Identity()
                    print("[*] Changing head for {} classes and removing aux classifier".format(num_classes))
                else:
                    self.model = deeplabv3_mobilenet_v3_large(pretrained_backbone=pretrained_backbone, num_classes=num_classes)
            elif backbone == "resnet101":
                if pretrained_head:
                    self.model= deeplabv3_resnet101(pretrained=True, pretrained_backbone=pretrained_backbone)
                    out_channel = 2048
                    if keep_features:
                        print("[*] Keeping features of deeplabv3, freezing all the model except the head")
                        self.freeze(1)
                    self.model.classifier = DeepLabHead(out_channel, num_classes)
                    self.model.aux_classifier = nn.Identity()
                    print("[*] Changing head for {} classes and removing aux classifier".format(num_classes))
                else:
                    self.model= deeplabv3_resnet101(pretrained_backbone=pretrained_backbone, num_classes=num_classes)
            else:
                assert "No such backbone"
            print("[*] Done")

        if self.alpha == 0 and keep_features == False:
            print("[*] Training the entire model with alpha {}".format(alpha))
            self.dofreeze = False
        elif self.dofreeze == True and keep_features == False:
            print("[!] This model will be trained using alpha freezing coef = {} meaning {}/{} layers will be freeze".format(self.alpha, int(self.alpha*sum(1 for x in self.model.parameters())), sum(1 for x in self.model.parameters())))
        
        if keep_features:
            return self.model, self.get_parameters()
        else:
            return self.model, self.model.parameters()
            
    def get_parameters(self) -> List:
        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
        return params_to_update
        
    def toString(self):
        for name, layer in self.model.named_modules():
            print(name, layer)

    def freeze(self, alpha: float=None):
        if self.dofreeze:
            s = sum(1 for x in self.model.parameters())
            if alpha == None:
                l_freeze = int(s*self.alpha)
            else:
                l_freeze = int(s*alpha)
            print("{} layers in this model, freezing {} layer\n".format(s, l_freeze))
            for i,param in enumerate(self.model.parameters()):
                param.requires_grad = False
                if l_freeze < i:
                    break
            #for name, layer in self.model.named_modules():
            #print(name, layer)
    
    def unfreeze(self):
        for i,param in enumerate(self.model.parameters()):
            param.requires_grad = True
