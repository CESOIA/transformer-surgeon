#!/bin/bash

# Path Configuration
# ======================
MODEL_PATH="pretrained/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="coco_indoor%10"                  # [DataArguments] Dataset with sampling rate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ARGS=$(cat <<EOF
--model_name_or_path $MODEL_PATH 
--tune_mm_llm True
--tune_mm_vision False 
--tune_mm_mlp False
--dataset_use $DATASETS 
--output_dir $OUTPUT_DIR 
--cache_dir $CACHE_DIR 
--bf16 true 
--per_device_train_batch_size 1 
--gradient_accumulation_steps 16 
--learning_rate 2e-7 
--mm_projector_lr 1e-5 
--vision_tower_lr 1e-6 
--optim adamw_torch 
--model_max_length 4096 
--data_flatten False
--data_packing False 
--base_interval 2 
--num_train_epochs 3 
--warmup_ratio 0.03 
--lr_scheduler_type cosine 
--weight_decay 0.01 
--logging_steps 10 
--save_steps 500 
--save_total_limit 3 
EOF
)

python qwenvl/train/train_qwen_c.py ${ARGS}

