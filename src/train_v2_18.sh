OUTPUT_DIR="sd-rgbd-model-lora-v2"
ACC_CONFIG="default_config"
CONFIG="/home/wukailu/.cache/huggingface/accelerate/$ACC_CONFIG.yaml"
MODEL_NAME="/home/wukailu/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"
DATA_DIR='/data/NYU_processed_disk/'
DATA_SPLIT="train+test"
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

#WANDB_MODE=offline
accelerate launch --config_file $CONFIG train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME\
  --local_data=$DATA_DIR --dataset_split=$DATA_SPLIT --caption_column="text" \
  --resolution=512 --random_flip --lora_rank=4\
  --train_batch_size=2 \
  --num_train_epochs=3 --checkpointing_steps=1000\
  --conv_learning_rate=1e-03 --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="bedroom made of cakes" --report_to="wandb" --enable_xformers_memory_efficient_attention\
  --resume_from_checkpoint latest --alpha_depth=0.8
