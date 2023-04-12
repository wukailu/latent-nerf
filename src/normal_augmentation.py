#!/usr/bin/env python
# coding: utf-8

# # import dataset from hugging face

# In[56]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
parser.add_argument('--st', type=int, default=0, help='continue from')
parser.add_argument('--en', type=int, default=0, help='end at (not included)')
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# # Augment the dataset by generate new images using depth image
# * 每张图片的GT深度图生成新的图片(5张)
# * 每张图片的GT法相生成新的图片(5张)


# In[ ]:


import sys
sys.path += ["/home/wukailu/latent-nerf/src/ControlNet"]
import einops
import numpy as np
import torch
import random
from PIL import Image
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# In[14]:


def process(detected_map, prompt, n_prompt, model, ddim_sampler, seed=-1, a_prompt='best quality, extremely detailed', num_samples=1, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, eta=0.0):
    with torch.no_grad():
        H, W, _ = detected_map.shape

        control = torch.from_numpy(detected_map.copy()).float().to(model.device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


# In[18]:

import cv2
import numpy as np

def get_surface_normal_by_depth(depth, K=None):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

def get_normal_map_by_point_cloud(depth, K):
    height, width = depth.shape

    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = (
        pts_3d_world[:, 1 : height - 1, 2:width]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    t = (
        pts_3d_world[:, 2:height, 1 : width - 1]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    normal_map = np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)
    return normal_map

# TODO: compute intrinsic of NYU_v2 using colmap
# vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
# normal1 = get_surface_normal_by_depth(depth, K)
# normal2 = get_normal_map_by_point_cloud(depth, K)


# In[49]:

negative_list = "text, signature, words, watermark, poster, postcards, username, faces, person, bodies, mutilated, " \
                "morbid, low quality, jpeg artifacts, duplicate, plane, cropped, worst quality, " \
                "signature, watermark, blurry, text, signature, words, watermark, poster, postcards, username"

if __name__ == "__main__":
    model = create_model('/home/wukailu/latent-nerf/src/ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/home/wukailu/latent-nerf/src/ControlNet/models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    save_memory = False

    print("work start!")
    gen_imgs = {}
    save_dir = "/data/NYU_processed/"
    save_interval = 1000
    cur = args.st
    skip_exist = True
    os.makedirs(save_dir + "gen_images", exist_ok=True)
    for image_id in range(args.st, args.en):
        disp_path = os.path.join(save_dir, "disparity", f"{image_id}.png")
        if os.path.exists(disp_path):
            if skip_exist and os.path.exists(save_dir + f"gen_images/{image_id}_gen_{1}.png"):
                continue
            print(f"mapping {image_id} to {args.gpu}")
            # Process the task
            disparity = np.array(Image.open(disp_path).convert("RGB"))
            ret = process(disparity, "indoor", negative_list, model, ddim_sampler,
                          num_samples=2, seed=2022211257)
            for idx, gen_img in enumerate(ret[1:]):
                Image.fromarray(gen_img).save(save_dir + f"gen_images/{image_id}_gen_{idx}.png")
