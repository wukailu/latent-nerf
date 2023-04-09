OUTPUT_DIR="sd-rgbd-model-lora"
MODEL_NAME="/home/wukailu/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"
DATA_DIR="/data/NYU_gen"
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

accelerate launch --num_processes=4 train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATA_DIR --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=20 --checkpointing_steps=10000 \
  --conv_learning_rate=1e-05 --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="bedroom made of cakes" --report_to="wandb"