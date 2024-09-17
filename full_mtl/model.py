import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN, MaskRCNN, KeypointRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

import numpy as np

class DetectionNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(DetectionNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses



class SegmentationNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(SegmentationNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses


class KeypointNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(KeypointNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses

class DenseposeNetwork(nn.Module):
    pass

class MTLModel(nn.Module):
    def __init__(self, pretrained=False, tasks = ['detection', 'segmentation', 'keypoints']) -> None:
        super(MTLModel, self).__init__()

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.preprocess = weights.transforms()


        if pretrained:
            self.weights = weights
        else:
            self.weights = None

        self.det_fullmodel = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        self.seg_fullmodel = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self.kp_fullmodel = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)

        self.transform = self.det_fullmodel.transform
        self.backbone = self.det_fullmodel.backbone

        if 'detection' in tasks:
            self.det_net = DetectionNetwork(self.det_fullmodel)
        else:
            self.det_net = None
        if 'segmentation' in tasks:
            self.seg_net = SegmentationNetwork(self.seg_fullmodel)
        else:
            self.seg_net = None
        if 'keypoints' in tasks:
            self.kp_net = KeypointNetwork(self.kp_fullmodel)
        else:
            self.kp_net = None


    def forward(self, x, targets=None):

        og_sizes = []

        for img in x:
            val = img.shape[-2:]
            og_sizes.append((val[0], val[1]))

        img_batch, x, img_sizes, targets = self._get_features(x, targets)
        det_output = self._faster_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.det_net else None
        seg_output = self._mask_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.seg_net else None
        kp_output = self._keypoint_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.kp_net else None

        return det_output, seg_output, kp_output

    def _get_features(self, x, targets=None):
        features, _ = self.transform(x)
        return features, self.backbone(features.tensors), features.image_sizes, targets

    def _faster_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.det_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)

    def _mask_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.seg_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)

    def _keypoint_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.kp_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)
