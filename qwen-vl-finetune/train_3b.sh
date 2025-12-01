#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="coco_indoor%10"                  # [DataArguments] Dataset with sampling rate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# ======================
# Training Hyperparameters
# ======================
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
--deepspeed scripts/zero3.json
EOF
)

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen_c.py ${ARGS}
