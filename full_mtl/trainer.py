import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from model import MTLModel
from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate


def train(model, train_loader, optimizer, device):
    model.train()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        det_losses, seg_losses, kp_losses = model(images, targets)
        print(det_losses, seg_losses, kp_losses)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tasks = ['detection', 'keypoint', 'segmentation']
    model = MTLModel(pretrained=True, tasks=tasks)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if 'keypoint' in tasks:
        dataset = CocoKeypointsDataset(datasetDir='../coco', split='train2017' )
    else:
        dataset = CocoInstancesDataset(datasetDir='../coco', split='train2017')
    train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, collate_fn=dataloader_collate)
    train(model, train_loader, optimizer, device)

if __name__ == "__main__":
    main()
