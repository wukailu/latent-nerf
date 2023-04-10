#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_from_disk
dataset = load_from_disk("/data/NYUDepthV2")


# # Augment the dataset by generate new images using depth image
# * 每张图片的GT深度图生成新的图片(5张)
# * 每张图片的GT法相生成新的图片(5张)


# In[ ]:


import sys
sys.path += ["/home/wukailu/ControlNet"]
import numpy as np
from PIL import Image

# In[18]:

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

if __name__ == "__main__":
    print("work start!")
    datalen = len(dataset["train"])
    for idx in tqdm(range(0, datalen)):
        depth_map, image_id = dataset["train"][idx]['depth_map'], idx
        image = np.array(dataset["train"][idx]['image'])[:, 80:-80]
        # Process the task
        disparity = depth_to_disparity(np.array(depth_map)[:, 80:-80])
        assert disparity.shape == (480, 480), str(disparity.shape)
        disparity = cv2.resize(disparity, (512, 512), interpolation=cv2.INTER_LINEAR)
        os.makedirs("/data/NYU_ori/disparity", exist_ok=True)
        os.makedirs("/data/NYU_ori/images", exist_ok=True)
        Image.fromarray(disparity).save(f"/data/NYU_ori/disparity/{image_id}.png")
        Image.fromarray(image).save(f"/data/NYU_ori/images/{image_id}.png")

