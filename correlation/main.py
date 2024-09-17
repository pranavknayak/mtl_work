import torch
import torch.nn as nn
import numpy as np

from torchvision.io import read_image

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

from dataset import CocoInstancesDataset

def main():
    det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    seg_weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1

    det_preprocess = det_weights.transforms()
    seg_preprocess = seg_weights.transforms()


    det_model = fasterrcnn_resnet50_fpn_v2(weights=det_weights, box_score_thresh=0.9)
    seg_model = maskrcnn_resnet50_fpn_v2(weights=seg_weights, box_score_thresh=0.9)

    dataDir = '../coco/'
    dataType = 'val2017'
    dataset = CocoInstancesDataset(dataDir, dataType)

    catCount = dataset.getCatLen()



    det_model.eval()
    seg_model.eval()
    for i in range(len(dataset)):
        img, targets = dataset[i]
        
        
        det_batch = det_preprocess(img).unsqueeze(0)
        seg_batch = seg_preprocess(img).unsqueeze(0)

        det_batch.requires_grad_()
        seg_batch.requires_grad_()
        
        noised_det_batch = det_batch + torch.randn_like(det_batch) * 0.01 # adding noise at a 1/100th the scale of pixel values
        noised_seg_batch = seg_batch + torch.randn_like(seg_batch) * 0.01 

        det_output = det_model(det_batch)[0]
        bboxes = det_output['boxes']
        det_labels = det_output['labels'].detach().numpy()

        noised_det_output = det_model(noised_det_batch)[0]
        noised_bboxes = noised_det_output['boxes']
        noised_det_labels = noised_det_output['labels'].detach().numpy()

        seg_output = seg_model(seg_batch)[0]
        masks = seg_output['masks']
        seg_labels = seg_output['labels'].detach().numpy()

        noised_seg_output = seg_model(noised_seg_batch)[0]
        noised_masks = noised_seg_output['masks']
        noised_seg_labels = noised_seg_output['labels'].detach().numpy()

        checked = []
        for label in targets['labels']:
            if label in checked:
                continue
            if label not in det_labels or label not in seg_labels:
                continue
            if label not in noised_det_labels or label not in noised_seg_labels:
                continue
            checked.append(label.item())

            det_label_idx = np.where(det_labels == label)
            seg_label_idx = np.where(seg_labels == label)

            det_bbox = bboxes[det_label_idx]
            seg_mask = masks[seg_label_idx]

            det_saliency = torch.autograd.grad(det_bbox.sum(), det_batch, retain_graph=True)[0]
            seg_saliency = torch.autograd.grad(seg_mask.sum(), seg_batch, retain_graph=True)[0]

            det_saliency = torch.flatten(det_saliency)
            seg_saliency = torch.flatten(seg_saliency)

            noised_det_label_idx = np.where(noised_det_labels == label)
            noised_seg_label_idx = np.where(noised_seg_labels == label)

            noised_det_bbox = noised_bboxes[noised_det_label_idx]
            noised_seg_mask = noised_masks[noised_seg_label_idx]

            noised_det_saliency = torch.autograd.grad(noised_det_bbox.sum(), noised_det_batch, retain_graph=True)[0]
            noised_seg_saliency = torch.autograd.grad(noised_seg_mask.sum(), noised_seg_batch, retain_graph=True)[0]

            noised_det_saliency = torch.flatten(noised_det_saliency)
            noised_seg_saliency = torch.flatten(noised_seg_saliency)

            det_sal_robustness = det_saliency - noised_det_saliency
            seg_sal_robustness = seg_saliency - noised_seg_saliency

            det_correlation = torch.dot(det_sal_robustness, seg_sal_robustness) / (torch.norm(det_sal_robustness) * torch.norm(seg_sal_robustness))
            print(f"DEBUG: {det_correlation}")
        
if __name__ == "__main__":
    main()
