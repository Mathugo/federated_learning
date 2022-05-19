# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layers for Unet model.
    Loss functions for semantic segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import keras.backend as K
import numpy as np

SMOOTH=1e-6

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def mIOU(pred, label, num_classes):
    pred = F.softmax(pred.float(), dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

#PyTorch Take low learning rate ~ 5e-5 1e-4
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class DiceBCELoss(nn.Module):
    def __init__(self, inputs, targets, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE    

class DiceLossMultiClass(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLossMultiClass, self).__init__() 

    def forward(self, inputs, targets, num_classes):
        dice_avg = 0
        print("\nInput shape {} Target Shape {}\n".format(inputs.shape, targets.shape))
        for index in range(num_classes):
            dice=DiceLoss().forward(inputs[ :, index, :, :], targets[ :, 0, :])
            print("Dice {} for class {} ".format(dice, index))
            dice_avg+=dice
            print("Total Dice {}".format(dice_avg))
        
        print("Dice avg {}".format(dice_avg/num_classes))
        return dice_avg/num_classes


#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MultiLossSegmentation(nn.Module):
    def __init__(self, binary_loss_function, weight=None, size_average=True):
        super(MultiLossSegmentation, self).__init__()
        self.binary_loss_function = binary_loss_function
    
    def forward(self, inputs, targets, num_classes, requires=True):
        print("\nInput shape {} Target Shape {}\n".format(inputs.shape, targets.shape))
        losses = torch.tensor([0], dtype=torch.float16)
        inputs = torch.sigmoid(inputs).to(torch.float32)  
        targets = targets.to(torch.float32)     
        #print("INPUTS \n"+str(inputs[ 0, 0, :, :]))
        #print("TARGETS \n"+str(targets[0, 0, :, :]))
        for index in range(num_classes):
            loss=self.binary_loss_function().forward(inputs[ :, index, :, :], targets[ :, 0, :, :])
            print("Loss for {} equals {} for class {} ".format(str(self.binary_loss_function), loss, index))
            losses +=loss
        #losses/=num_classes
        print("[!] TOTAL LOSS {}".format(losses))
        if requires:
            return losses.requires_grad_()
        else:
            return losses

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        print("\nInput shape {} Target Shape {}\n".format(inputs.shape, targets.shape))
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        print("INPUTS \n"+str(inputs[ 0, :, :, :]))
        print("TARGETS \n"+str(targets[0, :, :, :]))
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


#PyTorch Loss for class imbalanced
class BinaryFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

def soft_dice_loss(output, target):
    """Calculate loss."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score

# Dice evaluates the overlap rate of prediction results and ground truth; equals to f1 score in definition.
def soft_dice_coef(output, target):
    """Calculate soft DICE coefficient."""
    num = target.size(0)
    print("output {} target {}".format(output, target))
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.sum()

class DoubleConv(nn.Module):
    """Pytorch double conv class."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Do forward pass."""
        x = self.conv(x)
        return x

class Down(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        """Do forward pass."""
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch, bilinear=False):
        """Initialize layer."""
        super(Up, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """Do forward pass."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
