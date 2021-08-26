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

mask_path = "samples/0.png"
transform_mask_path = ""
image_path = "samples/0.jpg"

mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
image = cv2.imread(image_path)

cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

