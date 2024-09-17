import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2, keypointrcnn_resnet50_fpn

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, KeypointRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights

class RCNN_Backbone(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super(RCNN_Backbone, self).__init__()
        if pretrained:
            model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
            self.backbone = model.backbone
            self.transform = model.transform
        else:
            self.backbone = fasterrcnn_resnet50_fpn_v2().backbone
            self.transform = fasterrcnn_resnet50_fpn_v2().transform

    def forward(self, x):
        x = self.transform(x)
        x = x.tensors
        x = self.backbone(x)
        return x

def get_preprocessor():
    return FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

def resnet_backbone(pretrained=False):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        return resnet50(weights=weights)
    else:
        return resnet50()

def resnet_fpn_backbone(pretrained=False):
    if pretrained:
        return RCNN_Backbone(pretrained=True)
    else:
        return RCNN_Backbone(pretrained=False)

def detection_heads(pretrained=False):
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights
        faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=weights)
        return torch.nn.Sequential(faster_rcnn.rpn, faster_rcnn.roi_heads)
    else:
        faster_rcnn = fasterrcnn_resnet50_fpn_v2()
        return torch.nn.Sequential(faster_rcnn.rpn, faster_rcnn.roi_heads)

def segmentation_heads(pretrained=False):
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights
        maskrcnn = maskrcnn_resnet50_fpn_v2(weights=weights)
        return torch.nn.Sequential(maskrcnn.rpn, maskrcnn.roi_heads)
    else:
        maskrcnn = maskrcnn_resnet50_fpn_v2()
        return torch.nn.Sequential(maskrcnn.rpn, maskrcnn.roi_heads)

def keypoint_heads(pretrained=False):
    if pretrained:
        weights = KeypointRCNN_ResNet50_FPN_Weights
        keypointrcnn = keypointrcnn_resnet50_fpn(weights=weights)
        return torch.nn.Sequential(keypointrcnn.rpn, keypointrcnn.roi_heads)
    else:
        keypointrcnn = keypointrcnn_resnet50_fpn()
        return torch.nn.Sequential(keypointrcnn.rpn, keypointrcnn.roi_heads)

def faster_rcnn(pretrained=False):
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights
        return fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        return fasterrcnn_resnet50_fpn_v2()

def mask_rcnn(pretrained=False):
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights
        return maskrcnn_resnet50_fpn_v2(weights=weights)
    else:
        return maskrcnn_resnet50_fpn_v2()

def keypoint_rcnn(pretrained=False):
    if pretrained:
        weights = KeypointRCNN_ResNet50_FPN_Weights
        return keypointrcnn_resnet50_fpn(weights=weights)
    else:
        return keypointrcnn_resnet50_fpn()
