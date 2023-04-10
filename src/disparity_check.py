#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image

def depth_to_disparity(depth_map: np.array) -> np.array:
    valid_pixels = depth_map > 0
    disparity_map = np.zeros_like(depth_map, dtype=np.float32)
    disparity_map[valid_pixels] = 1.0 / depth_map[valid_pixels]
    min_disp = np.min(disparity_map[valid_pixels])
    max_disp = np.max(disparity_map[valid_pixels])
    disparity_map[valid_pixels] = (disparity_map[valid_pixels] - min_disp) / (max_disp - min_disp) * 255.
    disparity_map[depth_map == 0] = 0
    disparity_map = disparity_map.astype(np.uint8)
    return disparity_map


# In[49]:
import cv2
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    print("work start!")
    image_folder = "/data/NYU_ori/disparity"
    n_invalid = 0
    for img in tqdm(glob(image_folder + "/*.png")):
        disp = np.array(Image.open(img)).astype(np.float)
        if disp.min() != 0 or disp.max() != 255:
            valid_pixels = disp != 0
            min_disp = np.min(disp[valid_pixels])
            max_disp = np.max(disp[valid_pixels])
            disp[valid_pixels] = (disp[valid_pixels] - min_disp) / (max_disp - min_disp) * 255.
            disp[~valid_pixels] = 0
            disp = disp.astype(np.uint8)
            # Image.fromarray(disp).save(img)
            n_invalid += 1
    print("n_invalid: ", n_invalid, "  total:", len(glob(image_folder + "/*.png")))

