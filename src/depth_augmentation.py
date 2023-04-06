#!/usr/bin/env python
# coding: utf-8

# # import dataset from hugging face

# In[56]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
parser.add_argument('--ctn_from', type=int, default=0, help='continue from')
parser.add_argument('--total_gpu', type=int, default=8, help='number of total gpus')
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# from datasets import load_dataset
# dataset = load_dataset("sayakpaul/nyu_depth_v2")
# dataset.save_to_disk('/data/NYUDepthV2')

from datasets import load_from_disk
dataset = load_from_disk("/data/NYUDepthV2")


# # Augment the dataset by generate new images using depth image
# * 每张图片的GT深度图生成新的图片(5张)
# * 每张图片的GT法相生成新的图片(5张)


# In[ ]:


import sys
sys.path += ["/home/wukailu/ControlNet"]
import einops
import numpy as np
import torch
import random
from PIL import Image
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# In[14]:


def process(detected_map, prompt, n_prompt, model, ddim_sampler, seed=-1, a_prompt='best quality, extremely detailed', num_samples=1, ddim_steps=50, guess_mode=False, strength=1.0, scale=9.0, eta=0.0):
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

negative_list = "text, signature, words, watermark, poster, postcards, username, faces, person, bodies, mutilated, " \
                "morbid, low quality, jpeg artifacts, duplicate, plane, cropped, worst quality, " \
                "signature, watermark, blurry, text, signature, words, watermark, poster, postcards, username"

if __name__ == "__main__":
    model = create_model('/home/wukailu/ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/home/wukailu/ControlNet/models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    save_memory = False

    print("work start!")
    datalen = len(dataset["train"])
    for idx in range(args.gpu+(args.ctn_from//args.total_gpu*args.total_gpu), datalen, args.total_gpu):
        depth_map, image_id = dataset["train"][idx]['depth_map'], idx
        print(f"mapping {image_id} to {args.gpu}")
        # Process the task
        disparity = depth_to_disparity(np.array(depth_map))
        assert disparity.shape == (480, 640), str(disparity.shape)
        disparity = cv2.resize(disparity[:, 80:-80], (512, 512), interpolation=cv2.INTER_LINEAR)
        ret = process(disparity[..., np.newaxis].repeat(3, axis=-1), "indoor", negative_list, model, ddim_sampler,
                      num_samples=2, seed=2022211257)
        os.makedirs("/data/NYU_gen/disparity", exist_ok=True)
        os.makedirs("/data/NYU_gen/images", exist_ok=True)
        Image.fromarray(ret[0]).save(f"/data/NYU_gen/disparity/{image_id}.png")
        for idx, gen_img in enumerate(ret[1:]):
            Image.fromarray(gen_img).save(f"/data/NYU_gen/images/{image_id}_gen_{idx}.png")

