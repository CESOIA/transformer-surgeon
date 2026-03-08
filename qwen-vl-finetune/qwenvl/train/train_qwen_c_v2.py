import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class
import random


def split_dataset_to_disk(original_path, train_path, test_path, split_ratio=0.8, seed=42):
    """Splits the original JSON dataset into train and test parts on disk."""
    with open(original_path, "r") as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"[data] Dataset split completed: {len(train_data)} train, {len(test_data)} test")
    return len(train_data), len(test_data)


from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

from train_qwen import safe_save_model_for_hf_trainer, set_model

from math import ceil
from torch.utils.data import random_split

from transformers import TrainerCallback
import torch.distributed as dist

from test_messages import messages
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import torch.nn as nn

from pathlib import Path

class VCON_beta_upd(TrainerCallback):
    
    def __init__(self, manager, grad_acc_steps, vcon_grad_acc=0, beta=1.0):
        
        self.mgr = manager
        self.grad_acc_steps = grad_acc_steps
        self.beta = beta
        self.vcon_grad_acc = vcon_grad_acc
        self.step_cnt = 0
    
    # in case you want to update at the end of every training step
    # you should update beta after gradient_accumulation_steps 
    def on_step_end(self, args, state, control, **kwargs):
        
        if self.beta > 0:
            
            if self.step_cnt < self.grad_acc_steps:
                self.step_cnt = self.step_cnt + 1
            else:
                self._beta_upd()
                self.mgr.set_vcon_beta_all(self.beta)
                self.step_cnt = 0

    # When the number of steps in an epoch isn't divisible by grad_acc_steps
    def on_epoch_end(self, args, state, control, **kwargs):
        
        device = next(self.mgr.model.parameters()).device
        print("[trainer]: Model is on:", device)

        if self.beta > 0:
        
            if self.step_cnt: # if self.step_cnt is any value other than 0
                self._beta_upd()
                self.mgr.set_vcon_beta_all(self.beta)
                self.step_cnt = 0
                
    def _beta_upd(self):
         
        if self.vcon_grad_acc > 0:
            self.beta -= 1/self.vcon_grad_acc
        else:
            self.beta = 0

        if self.beta < 0:
            self.beta = 0


### COMPRESSION CONFIGURATION ###
model_type = "qwen2_5_vl_c" 
hard_mode = False
use_vcon = False  # Whether to use VCON blocks
vcon_beta = 1.0  # Beta value for VCON blocks (between 0 and 1)
vcon_epochs = 1 # For how many epochs will we apply VCON?
##########################

if model_type == "qwen2_vl_c":
    from transformersurgeon import (
        Qwen2VLForConditionalGenerationCompress,
        Qwen2VLConfigCompress,
        Qwen2VLCompressionSchemesManager,
    )

    modelClass = Qwen2VLForConditionalGenerationCompress
    configClass = Qwen2VLConfigCompress
    managerClass = Qwen2VLCompressionSchemesManager

    # Model name
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
elif model_type == "qwen2_5_vl_c":
    from transformersurgeon import (
        Qwen2_5_VLForConditionalGenerationCompress,
        Qwen2_5_VLConfigCompress,
        Qwen2_5_VLCompressionSchemesManager,
    )

    modelClass = Qwen2_5_VLForConditionalGenerationCompress
    configClass = Qwen2_5_VLConfigCompress
    managerClass = Qwen2_5_VLCompressionSchemesManager

    # Model name
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"


def train(attn_implementation="flash_attention_2"):
    
    # Load processor, model and tokenizer
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = modelClass.from_pretrained(model_name)

    device = next(model.parameters()).device
    print("Model is on:", device)
    
    mgr = managerClass(model)
    mgr.set("lrd", "rank", 128, [
        ["language_model", "mlp.down_proj"],
        ["language_model", "mlp.up_proj"],
        ["language_model", "mlp.gate_proj"]
    ], verbose=True)

    if use_vcon:
        mgr.init_vcon(verbose=True)
        mgr.set_vcon_beta(beta=vcon_beta)
    
    mgr.apply(verbose=True)
        
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    data_args.image_processor = processor.image_processor
    data_args.model_type = model_type
    
    
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    set_model(model_args, model)

    # if torch.distributed.get_rank() == 0:
        # model.model.print_trainable_parameters()

    # ---- Disk-based 80/20 train / test split ----
    original_dataset_path = data_args.dataset_use
    train_split_path = os.path.join("/ibex/user/antonic/CESOIA/datasets/COCO/annotations/", "train2017_qa_cats_chair_train.json")
    test_split_path = os.path.join("/ibex/user/antonic/CESOIA/datasets/COCO/annotations/", "train2017_qa_cats_chair_test.json")
    
    if training_args.local_rank in [-1, 0]:
        split_dataset_to_disk(original_dataset_path, train_split_path, test_split_path)
    
    # Wait for split to finish on rank 0
    if training_args.local_rank != -1:
        torch.distributed.barrier()
        
    data_args.dataset_use = "coco_qa_cats_chair_train" #train_split_path

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    train_dataset = data_module["train_dataset"]

    
    if use_vcon:
        
        nof_samples = len(data_module["train_dataset"])
        batch_size = training_args.per_device_train_batch_size
        nof_steps_in_epoch = ceil(nof_samples / batch_size)
        grad_acc_steps = training_args.gradient_accumulation_steps
        nof_grad_acc_in_epoch = ceil(nof_steps_in_epoch / grad_acc_steps)
        vcon_grad_acc = nof_grad_acc_in_epoch * vcon_epochs
        
        VCON_beta_upd_callback = VCON_beta_upd(mgr, grad_acc_steps=grad_acc_steps, vcon_grad_acc=vcon_grad_acc, beta=vcon_beta)
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module, callbacks=[VCON_beta_upd_callback]
        )
    else:

        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
        
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True
        
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
'''
    # ---- Collect responses on the test split ----
    if training_args.local_rank in [-1, 0]:
        print("\n[eval] Evaluating on the test split...")
        model.eval()
        
        test_split_path = os.path.join("/ibex/user/antonic/CESOIA/datasets/COCO/annotations/", "train2017_qa_cats_chair_test.json")
        with open(test_split_path, "r") as f:
            test_data = json.load(f)
            
        outputs = []
        for i, sample in enumerate(test_data):
            img_path = sample.get("image", "")
            img_fname = img_path
            
            img_path = os.path.join("/ibex/user/antonic/CESOIA/datasets/COCO/train2017/", img_fname)
            
            conversations = sample.get("conversations", [])
            if not conversations:
                continue
            
            instruction = conversations[0].get("value", "")
            ref_answer = conversations[1].get("value", "") if len(conversations) > 1 else ""
            
            # Format message for processor
            content = []
            if img_path:
                # Assuming images are in the current directory or paths are absolute
                # If they are relative to a specific dir, we might need a root path
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            
            # Process vision/text
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text_prompt],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
            ).to(model.device)
            
            # Inference
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
            # Trim prompt tokens
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = tokenizer.batch_decode(
                trimmed_ids, skip_special_tokens=True
            )[0].strip()
            
            # Requested format
            outputs.append({
                "img_fname": img_fname,
                "ref_answer": ref_answer,
                "model_ut_response": output_text
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
                print(f"  Processed {i+1}/{len(test_data)} test samples", end="\r")
        
        # Save to output_dir
        responses_path = os.path.join(training_args.output_dir, "test_eval_responses.json")
        with open(responses_path, "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"\n[eval] Saved {len(outputs)} test responses to {responses_path}")
    
    model.config.use_cache = True
'''


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

