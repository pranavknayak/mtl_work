from full_mtl.builder import FullMTL
from tooling.get_models import get_preprocessor
from torchvision.io import read_image

model = FullMTL()
img = read_image("../data/sample.jpg")
img = get_preprocessor()(img)
detections, masks, keypoints = model([img])
