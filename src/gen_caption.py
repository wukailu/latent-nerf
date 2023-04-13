#!/usr/bin/env python
# coding: utf-8

# # Convert augmented dataset into huggingface datasets ImageFolder format
# * For detail check https://huggingface.co/docs/datasets/v2.4.0/en/image_load#image-captioning
# * Containing RGB Image, Disparity, Text Captioning.
# In[]:
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default="A picture of,A styled room that,,A indoor room with",
                    help='comma seperated list of description prefix')
parser.add_argument('--question', type=str, default="",
                    help='comma seperated list of questions')
parser.add_argument('--data_folder', type=str, default="/data/NYU_processed/",
                    help='saved json lines config name')
parser.add_argument('--config_name', type=str, default="caption.jsonl",
                    help='saved json lines config name')
parser.add_argument('--img_template', type=str, default="images/*.png",
                    help='template of image path for glob, template will be <data_folder>/<img_template>')

args = parser.parse_args()

BLIP2_PATH = "Salesforce/blip2-opt-2.7b"
config_path = os.path.join(args.data_folder, args.config_name)

# In[1]:

import glob
import os

import jsonlines
if os.path.exists(config_path):
    with jsonlines.open(config_path, 'r') as reader:
        metadata = [obj for obj in reader]
else:
    metadata = []

filename_meta = {d["file_name"]: d['text'] for d in metadata}

# In[]:

import os
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from accelerate import Accelerator
from PIL import Image
import torch

accelerator = Accelerator()

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

processor = AutoProcessor.from_pretrained(BLIP2_PATH)
model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_PATH, device_map="auto", torch_dtype=weight_dtype)
processor, model = accelerator.prepare(processor, model)


# In[9]:

@torch.no_grad()
def get_caption(image: Image, prompt="A picture of "):
    inputs = processor(image, text=prompt, return_tensors="pt").to(dtype=weight_dtype)
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


# In[ ]:

from PIL import Image
from tqdm import tqdm

skip_processed = True
image_paths = glob.glob(os.path.join(args.data_folder, args.img_template))
for image_path in tqdm(image_paths, disable=not accelerator.is_local_main_process):
    file_name = os.path.relpath(image_path, args.data_folder)
    if (file_name in filename_meta) and skip_processed:
        continue
    image = Image.open(image_path).convert("RGB")
    prefix_list = args.prefix.split(",")
    question_list = [] if len(args.question) == 0 else args.question.split(",")
    caption = [prefix + " " + get_caption(image, prefix) for prefix in prefix_list]
    caption += [get_caption(image, f"Question: {question} Answer:") for question in question_list]
    info = {
        "file_name": file_name,
        "text": caption,
    }
    metadata.append(info)
    with jsonlines.open(config_path, 'a') as writer:
        writer.write(info)

with jsonlines.open(config_path, 'w') as writer:
    writer.write_all(metadata)
