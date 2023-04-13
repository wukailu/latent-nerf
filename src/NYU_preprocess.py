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

# In[ ]:

# Fill depth

def preprocess(d):
    H, W = d.shape[:2]

    d[d == 65535] = 0  # invalid values

    loops = 0
    while d.min() == 0:
        loops += 1
        if loops > 1000:
            assert False, "impossible to fill depth img"
        idx_0, idx_1 = np.where(d == 0)
        d_fill = np.zeros(d.shape)
        d_fill[idx_0, idx_1] = 1

        for i in range(H):
            y_idx = np.where(d_fill[i] > 0)[0]

            if len(y_idx) == 0: continue
            if len(y_idx) == 1:
                d_fill[i, y_idx[0]] = (d[i, y_idx[0] - 1] + d[i, (y_idx[0] + 1) % W]) / 2
                continue
            if len(y_idx) == W:
                d_fill[i] = 0
                if i != 0 and d[i - 1, 0] != 0:
                    d[i, 0] = d[i - 1, 0]
                else:
                    d[i, 0] = d[min(i + 1, H - 1), 0]
                continue

            gaps = [[s, e] for s, e in zip(y_idx, y_idx[1:]) if s + 1 < e]
            edges = np.concatenate([y_idx[:1], np.array(sum(gaps, [])), y_idx[-1:]])

            interval = [[int(s), int(e) + 1] for s, e in zip(edges[::2], edges[1:][::2])]
            if interval[0][0] == 0:
                interval[0][0] = interval[-1][0] - W
                interval = interval[:-1]

            for s, e in interval:
                if s < 0:
                    interp = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
                    d_fill[i, s:] = interp[:-s]
                    d_fill[i, :e] = interp[-s:]
                else:
                    d_fill[i, s:e] = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
        d = d + d_fill
    return d

# In[18]:

def depth_to_disparity(depth_map: np.array) -> np.array:
    valid_pixels = depth_map > 0
    disparity_map = np.zeros_like(depth_map, dtype=np.float32)
    disparity_map[valid_pixels] = 1.0 / depth_map[valid_pixels]
    min_disp = 0.0
    max_disp = np.max(disparity_map[valid_pixels])
    disparity_map[valid_pixels] = (disparity_map[valid_pixels] - min_disp) / (max_disp - min_disp) * 255.
    disparity_map[~valid_pixels] = 0
    return disparity_map.round().astype(np.uint8)

# In[]:
def process_item(idx):
    depth_map, image_id = np.array(dataset["train"][idx]['depth_map']), idx
    depth_map = preprocess(depth_map)[10:-10, 90:-90]
    depth_map = cv2.resize(depth_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    disparity_map = depth_to_disparity(depth_map)
    image = np.array(dataset["train"][idx]['image'])[10:-10, 90:-90]
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC).round().astype(np.uint8)
    os.makedirs(save_dir + "disparity", exist_ok=True)
    os.makedirs(save_dir + "origin_images", exist_ok=True)
    Image.fromarray(disparity_map).save(os.path.join(save_dir, "disparity", f"{image_id}.png"))
    Image.fromarray(image).save(os.path.join(save_dir, "origin_images", f"{image_id}.png"))
    return depth_map, disparity_map, image

# In[49]:
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == "__main__":
    print("work start!")
    save_dir = "/data/NYU_processed/"
    datalen = len(dataset["train"])
    ret = Parallel(n_jobs=96)(delayed(process_item)(idx) for idx in tqdm(range(0, datalen)))
    depth_maps, disparity_maps, cropped_images = list(zip(*ret))

    import pickle
    with open(save_dir + "data.pkl", "wb") as f:
        pickle.dump({
            "depth": depth_maps,
            "disparity": depth_maps,
            "image": cropped_images,
        }, f)
