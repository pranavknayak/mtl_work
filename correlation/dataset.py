import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class CocoInstancesDataset(Dataset):
    def __init__(self, datasetDir: str, split: str) -> None:
        super().__init__()
        self.dataDir = f'{datasetDir}/images/{split}'

        self.annFile = f'{datasetDir}/annotations/instances_{split}.json'
        self._coco = COCO(self.annFile)

        self.imgIds = self._coco.getImgIds()

    def _xywh_to_xyxy(self, box):
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x2 = x + w
        y2 = y + h 
        return [x, y, x2, y2]

    def __len__(self) -> int:
        return len(self.imgIds)

    def __getitem__(self, idx: int):
        img_id = self.imgIds[idx]
        img_obj = self._coco.loadImgs(img_id)[0]

        annotation = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))
        img = Image.open(f'{self.dataDir}/{img_obj["file_name"]}')
        bboxes = [self._xywh_to_xyxy(ann['bbox']) for ann in annotation]
        masks = np.array([self._coco.annToMask(ann) for ann in annotation])
        areas = [ann['area'] for ann in annotation]

        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor([ann['category_id'] for ann in annotation], dtype=torch.int64),
            'masks': torch.tensor(masks, dtype=torch.float32),
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor([ann['iscrowd'] for ann in annotation], dtype=torch.int64)
        }

        return img, target

    def getCatLen(self):
        return len(self._coco.getCatIds())
