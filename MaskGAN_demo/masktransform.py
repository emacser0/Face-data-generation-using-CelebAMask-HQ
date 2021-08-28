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

image_directory = "../face_parsing/Data_preprocessing/test_img/"
mask_directory = "../face_parsing/test_results/"

result_image_directory = "AdversarialSample/image/"
result_mask_directory = "AdversarialSample/mask/"

image_extension = ".jpg"
mask_extension = ".png"

move_ranges = [(-100, -50), (50, 100)]
move_features = [NOSE, LEFTEYE, RIGHTEYE, UPPERLIP, LEFTBROW, RIGHTBROW]

scale_ranges = [(0.2, 0.5), (1.5, 2.0)]
scale_features = [NOSE, LEFTEYE, RIGHTEYE, UPPERLIP, LEFTBROW, RIGHTBROW]

def get_bounding_box(mask):
    boxes = [((mask.shape[1], mask.shape[0]), (0, 0)) for _ in range(19)]
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            c = int(mask[i, j, 0])
            ltx, lty = boxes[c][0]
            rbx, rby = boxes[c][1]
            boxes[c] = ((min(ltx, j), min(lty, i)), (max(rbx, j), max(rby, i)))
    return boxes

def draw_bounding_box(mask, image, normalize=False):
    boxes = get_bounding_box(mask)

    for index, box in enumerate(boxes):
        if index < 2:
            continue
        if normalize:
            cv2.rectangle(image, box[0], box[1], (0, index / len(boxes) * 255, 0), 2)
        else:
            cv2.rectangle(image, box[0], box[1], (0, index, 0), 2)
    return image

def move_mask(mask, index, dx, dy, fill=0):
    result = mask.copy()
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[i, j, 0] == index:
                if i + dy >= mask.shape[0] or j + dx >= mask.shape[1]:
                    continue
                result[i + dy, j + dx] = mask[i, j]
                result[i, j] = fill
    return result

def generate_adversarial_sample(model, image_path, mask_path):
    mask = cv2.imread(mask_path)
    image = Image.open(image_path)

    mask_m = mask.copy()
    move_feature_count = min(len(move_features) - 1, round(abs(np.random.normal(0, 1))) + 1)
    print(move_feature_count)
    feature_indexes = np.random.choice(range(len(move_features)), move_feature_count)
    for feature_index in feature_indexes:
        feature_index = np.random.randint(0, len(move_features))

        x_range_index = np.random.randint(0, len(move_ranges))
        y_range_index = np.random.randint(0, len(move_ranges))

        x_lower_bound, x_upper_bound = move_ranges[x_range_index]
        y_lower_bound, y_upper_bound = move_ranges[y_range_index]

        offset_x = np.random.randint(x_lower_bound, x_upper_bound + 1)
        offset_y = np.random.randint(y_lower_bound, y_upper_bound + 1)

        mask_m = move_mask(mask_m, move_features[feature_index], offset_x, offset_y, fill=SKIN)
        if move_features[feature_index] == UPPERLIP:
            mask_m = move_mask(mask_m, LOWERLIP, offset_x, offset_y, fill=SKIN)
            mask_m = move_mask(mask_m, MOUTH, offset_x, offset_y, fill=SKIN)

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

    #result = draw_bounding_box(mask_m, result, normalize=True)
    return (result, mask_m)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)

    params = get_params(opt, (1024, 1024))

    sample_count = 30000
    iteration_count = 1000

    os.makedirs(result_image_directory, exist_ok=True)
    os.makedirs(result_mask_directory, exist_ok=True)

    for i in range(iteration_count):
        sample_index = np.random.randint(0, sample_count)
        image_path = image_directory + str(sample_index) + image_extension
        mask_path = mask_directory + str(sample_index) + mask_extension

        result_image, result_mask = generate_adversarial_sample(model, image_path, mask_path)

        cv2.imwrite(result_image_directory + str(i) + image_extension, result_image)
        cv2.imwrite(result_mask_directory + str(i) + mask_extension, result_mask)