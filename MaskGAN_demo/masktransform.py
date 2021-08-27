import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize

from enum import Enum

BACKGROUND = 0
SKIN = 1
NOSE = 2
EYEGLASSES = 3
LEFTEYE = 4
RIGHTEYE = 5
LEFTBROW = 6
RIGHTBROW = 7
LEFTEAR = 8
RIGHTEAR = 9
MOUTH = 10
UPPERLIP = 11
LOWERLIP = 12
HAIR = 13
HAT = 14
EARRING = 15
NECKLACE = 16
NECK = 17
CLOTH = 18

mask_path = "../face_parsing/test_results/0.png"
image_path = "../face_parsing/Data_preprocessing/test_img/0.jpg"

mask = cv2.imread(mask_path)
image = Image.open(image_path)

def get_bounding_box_from_mask(mask):
    boxes = [((mask.shape[1], mask.shape[0]), (0, 0)) for _ in range(19)]
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            c = int(mask[i, j, 0])
            ltx, lty = boxes[c][0]
            rbx, rby = boxes[c][1]
            boxes[c] = ((min(ltx, j), min(lty, i)), (max(rbx, j), max(rby, i)))
    return boxes

def draw_bounding_box(mask, image, normalize=False):
    boxes = get_bounding_box_from_mask(mask)

    for index, box in enumerate(boxes):
        if index < 2:
            continue
        if normalize:
            cv2.rectangle(image, box[0], box[1], (0, index / len(boxes) * 255, 0), 2)
        else:
            cv2.rectangle(image, box[0], box[1], (0, index, 0), 2)
    return image

def translate_mask(mask, index, dx, dy, fill=0):
    result = mask.copy()
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[i, j, 0] == index:
                result[i + dy, j + dx] = mask[i, j]
                result[i, j] = fill
    return result

mask_m = translate_mask(mask, LEFTEYE, 0, 100, fill=SKIN)
mask_m = translate_mask(mask_m, RIGHTEYE, 0, -100, fill=SKIN)
mask_m = translate_mask(mask_m, UPPERLIP, 0, -20, fill=SKIN)
mask_m = translate_mask(mask_m, LOWERLIP, 0, -20, fill=SKIN)
'''
mask_m = (mask_m / mask_m.max()) * 255
mask_m = np.array(mask_m, dtype=np.uint8)
mask_m = cv2.applyColorMap(mask_m, cv2.COLORMAP_TWILIGHT)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
model = create_model(opt)

params = get_params(opt, (1024, 1024))

transform_mask = get_transform(opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
transform_image = get_transform(opt, params)

mask_t = transform_mask(Image.fromarray(np.uint8(mask)))
mask_m_t = transform_mask(Image.fromarray(np.uint8(mask_m)))
image_t = transform_image(Image.fromarray(np.uint8(image)))

generated = model.inference(torch.FloatTensor([mask_m_t.numpy()]), torch.FloatTensor([mask_t.numpy()]), torch.FloatTensor([image_t.numpy()]))

result = generated.permute(0, 2, 3, 1)
result = result.cpu().detach().numpy()
result = (result + 1) * 127.5
result = np.asarray(result[0,:,:,:], dtype=np.uint8)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

result = draw_bounding_box(mask_m, result, normalize=True)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()