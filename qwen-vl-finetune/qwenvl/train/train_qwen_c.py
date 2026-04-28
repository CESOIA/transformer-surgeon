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

from transformers import TrainerCallback
import torch.distributed as dist

from test_messages import messages
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import torch.nn as nn


class VCON_cancel(TrainerCallback):
    """Cancels VCON blocks at the end of the full training loop."""

    def __init__(self, manager):
        self.mgr = manager

    def on_train_end(self, args, state, control, **kwargs):
        print("[trainer]: Training complete — cancelling VCON blocks.")
        self.mgr.cancel_vcon()
        print("[trainer]: VCON blocks cancelled successfully.")


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
            
            if self.step_cnt < self.grad_acc_steps-1:
                self.step_cnt = self.step_cnt + 1
            else:
                self._beta_upd()
                self.mgr.set_vcon_beta(self.beta)
                self.step_cnt = 0

    # When the number of steps in an epoch isn't divisible by grad_acc_steps
    def on_epoch_end(self, args, state, control, **kwargs):
        
        device = next(self.mgr.model.parameters()).device
        print("[trainer]: Model is on:", device)

        if self.beta > 0:
        
            if self.step_cnt: # if self.step_cnt is any value other than 0
                self._beta_upd()
                self.mgr.set_vcon_beta(self.beta)
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
vcon_epochs = 2 # For how many epochs will we apply VCON?
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
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"


def train(attn_implementation="flash_attention_2"):
    
    # Load processor, model and tokenizer
	processor = AutoProcessor.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
    
	model = modelClass.from_pretrained(model_name)

	device = next(model.parameters()).device
	print("Model is on:", device)
    
	mgr = managerClass(model)

	# =========================
	# LRD ranks for all layers
	# =========================
	
	# rank = 1638

	mgr.set("lrd", "rank", 1638, [
		["language_model.layers.17", "mlp.up_proj"],
		["language_model.layers.28", "mlp.down_proj"],
		["language_model.layers.31", "mlp.down_proj"],
		["language_model.layers.32", "mlp.down_proj"],
		["language_model.layers.34", "mlp.down_proj"],
		["language_model.layers.35", "mlp.down_proj"]
	], verbose=True)
	
	# rank = 1536
	mgr.set("lrd", "rank", 1536, [
		["language_model.layers.31", "mlp.gate_proj"],
		["language_model.layers.33", "mlp.up_proj"],
		["language_model.layers.34", "mlp.gate_proj"]
	], verbose=True)

	# rank = 1434
	mgr.set("lrd", "rank", 1434, [
		["language_model.layers.11", "mlp.down_proj"],
		["language_model.layers.9", "mlp.down_proj"],
		["language_model.layers.30", "mlp.gate_proj"],
		["language_model.layers.32", "mlp.gate_proj"],
		["language_model.layers.35", "mlp.gate_proj"]
	], verbose=True)

	# rank = 1331
	mgr.set("lrd", "rank", 1331, [
		["language_model.layers.0", "mlp.down_proj"],
		["language_model.layers.0", "mlp.gate_proj"],
		["language_model.layers.11", "mlp.gate_proj"],
		["language_model.layers.17", "mlp.gate_proj"],
		["language_model.layers.18", "mlp.up_proj"],
		["language_model.layers.22", "mlp.up_proj"],
		["language_model.layers.23", "mlp.down_proj"],
		["language_model.layers.29", "mlp.down_proj"],
		["language_model.layers.30", "mlp.down_proj"],
		["language_model.layers.32", "mlp.up_proj"],
		["language_model.layers.33", "mlp.down_proj"],
		["language_model.layers.4", "mlp.down_proj"],
		["language_model.layers.5", "mlp.up_proj"]
	], verbose=True)

	# rank = 1229
	mgr.set("lrd", "rank", 1229, [
		["language_model.layers.0", "mlp.up_proj"],
		["language_model.layers.10", "mlp.up_proj"],
		["language_model.layers.14", "mlp.gate_proj"],
		["language_model.layers.16", "mlp.down_proj"],
		["language_model.layers.20", "mlp.up_proj"],
		["language_model.layers.24", "mlp.gate_proj"],
		["language_model.layers.24", "mlp.up_proj"],
		["language_model.layers.27", "mlp.down_proj"],
		["language_model.layers.28", "mlp.gate_proj"],
		["language_model.layers.35", "mlp.up_proj"],
		["language_model.layers.5", "mlp.down_proj"],
		["language_model.layers.7", "mlp.up_proj"],
		["language_model.layers.8", "mlp.gate_proj"],
		["language_model.layers.9", "mlp.up_proj"]
	], verbose=True)

	# rank = 1126
	mgr.set("lrd", "rank", 1126, [
		["language_model.layers.13", "mlp.down_proj"],
		["language_model.layers.13", "mlp.up_proj"],
		["language_model.layers.14", "mlp.up_proj"],
		["language_model.layers.16", "mlp.gate_proj"],
		["language_model.layers.17", "mlp.down_proj"],
		["language_model.layers.21", "mlp.gate_proj"],
		["language_model.layers.22", "mlp.down_proj"],
		["language_model.layers.26", "mlp.down_proj"],
		["language_model.layers.29", "mlp.gate_proj"],
		["language_model.layers.30", "mlp.up_proj"],
		["language_model.layers.3", "mlp.gate_proj"],
		["language_model.layers.4", "mlp.gate_proj"],
		["language_model.layers.4", "mlp.up_proj"],
		["language_model.layers.5", "mlp.gate_proj"],
		["language_model.layers.6", "mlp.gate_proj"],
		["language_model.layers.7", "mlp.down_proj"],
		["language_model.layers.7", "mlp.gate_proj"],
		["language_model.layers.8", "mlp.down_proj"],
		["language_model.layers.8", "mlp.up_proj"]
	], verbose=True)

	# rank = 1024
	mgr.set("lrd", "rank", 1024, [
		["language_model.layers.11", "mlp.up_proj"],
		["language_model.layers.12", "mlp.gate_proj"],
		["language_model.layers.15", "mlp.gate_proj"],
		["language_model.layers.16", "mlp.up_proj"],
		["language_model.layers.19", "mlp.down_proj"],
		["language_model.layers.22", "mlp.gate_proj"],
		["language_model.layers.24", "mlp.down_proj"],
		["language_model.layers.26", "mlp.gate_proj"],
		["language_model.layers.27", "mlp.up_proj"],
		["language_model.layers.29", "mlp.up_proj"],
		["language_model.layers.3", "mlp.up_proj"],
		["language_model.layers.34", "mlp.up_proj"],
#		["language_model.layers.35", "self_attn.o_proj"],
#		["language_model.layers.35", "self_attn.q_proj"],
		["language_model.layers.6", "mlp.down_proj"],
		["language_model.layers.6", "mlp.up_proj"],
#		["language_model.layers.6", "self_attn.o_proj"],
#		["language_model.layers.7", "self_attn.o_proj"]
	], verbose=True)

	# rank = 922
	mgr.set("lrd", "rank", 922, [
		["language_model.layers.1", "mlp.down_proj"],
		["language_model.layers.1", "mlp.gate_proj"],
		["language_model.layers.10", "mlp.down_proj"],
		["language_model.layers.10", "mlp.gate_proj"],
		["language_model.layers.12", "mlp.down_proj"],
		["language_model.layers.12", "mlp.up_proj"],
		["language_model.layers.13", "mlp.gate_proj"],
		["language_model.layers.14", "mlp.down_proj"],
		["language_model.layers.15", "mlp.down_proj"],
		["language_model.layers.18", "mlp.down_proj"],
		["language_model.layers.18", "mlp.gate_proj"],
		["language_model.layers.19", "mlp.gate_proj"],
		["language_model.layers.20", "mlp.gate_proj"],
		["language_model.layers.21", "mlp.down_proj"],
		["language_model.layers.25", "mlp.down_proj"],
		["language_model.layers.26", "mlp.up_proj"],
		["language_model.layers.28", "mlp.up_proj"],
		["language_model.layers.3", "mlp.down_proj"],
		["language_model.layers.31", "mlp.up_proj"],
		["language_model.layers.9", "mlp.gate_proj"]

#		["language_model.layers.12", "self_attn.o_proj"],
#		["language_model.layers.15", "self_attn.o_proj"],
#		["language_model.layers.17", "self_attn.o_proj"],
#		["language_model.layers.2", "self_attn.q_proj"],
#		["language_model.layers.21", "self_attn.q_proj"],
#		["language_model.layers.22", "self_attn.o_proj"],
#		["language_model.layers.26", "self_attn.q_proj"],
#		["language_model.layers.27", "self_attn.o_proj"],
#		["language_model.layers.27", "self_attn.q_proj"],
#		["language_model.layers.29", "self_attn.o_proj"],
#		["language_model.layers.3", "self_attn.o_proj"],
#		["language_model.layers.34", "self_attn.q_proj"],
#		["language_model.layers.4", "self_attn.o_proj"],
#		["language_model.layers.4", "self_attn.q_proj"],
#		["language_model.layers.5", "self_attn.q_proj"],
#		["language_model.layers.6", "self_attn.q_proj"],
#		["language_model.layers.9", "self_attn.o_proj"]
	], verbose=True)

	# rank = 819
	mgr.set("lrd", "rank", 819, [
		["language_model.layers.19", "mlp.up_proj"],
		["language_model.layers.2", "mlp.down_proj"],
		["language_model.layers.2", "mlp.up_proj"],
		["language_model.layers.21", "mlp.up_proj"],
		["language_model.layers.23", "mlp.gate_proj"],
		["language_model.layers.23", "mlp.up_proj"],
		["language_model.layers.25", "mlp.gate_proj"]

#		["language_model.layers.1", "self_attn.q_proj"],
#		["language_model.layers.14", "self_attn.o_proj"],
#		["language_model.layers.18", "self_attn.q_proj"],
#		["language_model.layers.21", "self_attn.o_proj"],
#		["language_model.layers.22", "self_attn.q_proj"],
#		["language_model.layers.24", "self_attn.o_proj"],
#		["language_model.layers.24", "self_attn.q_proj"],
#		["language_model.layers.3", "self_attn.q_proj"],
#		["language_model.layers.31", "self_attn.q_proj"],
#		["language_model.layers.34", "self_attn.o_proj"],
#		["language_model.layers.9", "self_attn.q_proj"],
#		["language_model.layers.2", "self_attn.o_proj"],
#		["language_model.layers.5", "self_attn.o_proj"]
	], verbose=True)

	# rank = 717
	mgr.set("lrd", "rank", 717, [
		["language_model.layers.1", "mlp.up_proj"],
		["language_model.layers.20", "mlp.down_proj"],
		["language_model.layers.25", "mlp.up_proj"]

#		["language_model.layers.11", "self_attn.o_proj"],
#		["language_model.layers.11", "self_attn.q_proj"],
#		["language_model.layers.15", "self_attn.q_proj"],
#		["language_model.layers.19", "self_attn.q_proj"],
#		["language_model.layers.23", "self_attn.q_proj"],
#		["language_model.layers.28", "self_attn.o_proj"],
#		["language_model.layers.7", "self_attn.q_proj"]
	], verbose=True)

	# rank = 614
#	mgr.set("lrd", "rank", 614, [
#		["language_model.layers.20", "self_attn.o_proj"],
#		["language_model.layers.25", "self_attn.q_proj"],
#		["language_model.layers.30", "self_attn.q_proj"],
#		["language_model.layers.8", "self_attn.o_proj"],
#		["language_model.layers.8", "self_attn.q_proj"]
#	], verbose=True)

	# rank = 512
#	mgr.set("lrd", "rank", 512, [
#		["language_model.layers.20", "self_attn.q_proj"],
#		["language_model.layers.33", "self_attn.q_proj"]
#	], verbose=True)

	# =========================
	# Attention k_proj / v_proj ranks
	# keeping only final_rank < 128
	# =========================
	'''
	# rank = 115
	mgr.set("lrd", "rank", 115, [
		["language_model.layers.10", "self_attn.v_proj"],
		["language_model.layers.11", "self_attn.k_proj"],
		["language_model.layers.11", "self_attn.v_proj"],
		["language_model.layers.22", "self_attn.v_proj"],
		["language_model.layers.23", "self_attn.v_proj"],
		["language_model.layers.29", "self_attn.v_proj"],
		["language_model.layers.35", "self_attn.k_proj"],
		["language_model.layers.6", "self_attn.k_proj"]
	], verbose=True)

	# rank = 102
	mgr.set("lrd", "rank", 102, [
		["language_model.layers.1", "self_attn.k_proj"],
		["language_model.layers.12", "self_attn.k_proj"],
		["language_model.layers.15", "self_attn.v_proj"],
		["language_model.layers.2", "self_attn.k_proj"],
		["language_model.layers.20", "self_attn.v_proj"],
		["language_model.layers.24", "self_attn.v_proj"],
		["language_model.layers.25", "self_attn.v_proj"],
		["language_model.layers.28", "self_attn.v_proj"],
		["language_model.layers.3", "self_attn.k_proj"],
		["language_model.layers.31", "self_attn.v_proj"],
		["language_model.layers.33", "self_attn.k_proj"],
		["language_model.layers.9", "self_attn.k_proj"]
	], verbose=True)

	# rank = 90
	mgr.set("lrd", "rank", 90, [
		["language_model.layers.12", "self_attn.v_proj"],
		["language_model.layers.16", "self_attn.k_proj"],
		["language_model.layers.19", "self_attn.k_proj"],
		["language_model.layers.20", "self_attn.k_proj"],
		["language_model.layers.4", "self_attn.k_proj"],
		["language_model.layers.7", "self_attn.k_proj"]
	], verbose=True)

	# rank = 77
	mgr.set("lrd", "rank", 77, [
		["language_model.layers.17", "self_attn.v_proj"],
		["language_model.layers.2", "self_attn.v_proj"]
	], verbose=True)

	# rank = 64
	mgr.set("lrd", "rank", 64, [
		["language_model.layers.27", "self_attn.k_proj"]
	], verbose=True)

	# rank = 51
	mgr.set("lrd", "rank", 51, [
		["language_model.layers.13", "self_attn.k_proj"]
	], verbose=True)
	'''
	# =========================
	# VCON for MLP + attention
	# =========================
	if use_vcon:
		mlp_criteria = [
		    ["language_model.layers.0", "mlp.down_proj"],
		    ["language_model.layers.0", "mlp.gate_proj"],
		    ["language_model.layers.0", "mlp.up_proj"],

		    ["language_model.layers.1", "mlp.down_proj"],
		    ["language_model.layers.1", "mlp.gate_proj"],
		    ["language_model.layers.1", "mlp.up_proj"],

		    ["language_model.layers.2", "mlp.down_proj"],
		    ["language_model.layers.2", "mlp.gate_proj"],
		    ["language_model.layers.2", "mlp.up_proj"],

		    ["language_model.layers.3", "mlp.down_proj"],
		    ["language_model.layers.3", "mlp.gate_proj"],
		    ["language_model.layers.3", "mlp.up_proj"],

		    ["language_model.layers.4", "mlp.down_proj"],
		    ["language_model.layers.4", "mlp.gate_proj"],
		    ["language_model.layers.4", "mlp.up_proj"],

		    ["language_model.layers.5", "mlp.down_proj"],
		    ["language_model.layers.5", "mlp.gate_proj"],
		    ["language_model.layers.5", "mlp.up_proj"],

		    ["language_model.layers.6", "mlp.down_proj"],
		    ["language_model.layers.6", "mlp.gate_proj"],
		    ["language_model.layers.6", "mlp.up_proj"],

		    ["language_model.layers.7", "mlp.down_proj"],
		    ["language_model.layers.7", "mlp.gate_proj"],
		    ["language_model.layers.7", "mlp.up_proj"],

		    ["language_model.layers.8", "mlp.down_proj"],
		    ["language_model.layers.8", "mlp.gate_proj"],
		    ["language_model.layers.8", "mlp.up_proj"],

		    ["language_model.layers.9", "mlp.down_proj"],
		    ["language_model.layers.9", "mlp.gate_proj"],
		    ["language_model.layers.9", "mlp.up_proj"],

		    ["language_model.layers.10", "mlp.down_proj"],
		    ["language_model.layers.10", "mlp.gate_proj"],
		    ["language_model.layers.10", "mlp.up_proj"],

		    ["language_model.layers.11", "mlp.down_proj"],
		    ["language_model.layers.11", "mlp.gate_proj"],
		    ["language_model.layers.11", "mlp.up_proj"],

		    ["language_model.layers.12", "mlp.down_proj"],
		    ["language_model.layers.12", "mlp.gate_proj"],
		    ["language_model.layers.12", "mlp.up_proj"],

		    ["language_model.layers.13", "mlp.down_proj"],
		    ["language_model.layers.13", "mlp.gate_proj"],
		    ["language_model.layers.13", "mlp.up_proj"],

		    ["language_model.layers.14", "mlp.down_proj"],
		    ["language_model.layers.14", "mlp.gate_proj"],
		    ["language_model.layers.14", "mlp.up_proj"],

		    ["language_model.layers.15", "mlp.down_proj"],
		    ["language_model.layers.15", "mlp.gate_proj"],
		    ["language_model.layers.15", "mlp.up_proj"],

		    ["language_model.layers.16", "mlp.down_proj"],
		    ["language_model.layers.16", "mlp.gate_proj"],
		    ["language_model.layers.16", "mlp.up_proj"],

		    ["language_model.layers.17", "mlp.down_proj"],
		    ["language_model.layers.17", "mlp.gate_proj"],
		    ["language_model.layers.17", "mlp.up_proj"],

		    ["language_model.layers.18", "mlp.down_proj"],
		    ["language_model.layers.18", "mlp.gate_proj"],
		    ["language_model.layers.18", "mlp.up_proj"],

		    ["language_model.layers.19", "mlp.down_proj"],
		    ["language_model.layers.19", "mlp.gate_proj"],
		    ["language_model.layers.19", "mlp.up_proj"],

		    ["language_model.layers.20", "mlp.down_proj"],
		    ["language_model.layers.20", "mlp.gate_proj"],
		    ["language_model.layers.20", "mlp.up_proj"],

		    ["language_model.layers.21", "mlp.down_proj"],
		    ["language_model.layers.21", "mlp.gate_proj"],
		    ["language_model.layers.21", "mlp.up_proj"],

		    ["language_model.layers.22", "mlp.down_proj"],
		    ["language_model.layers.22", "mlp.gate_proj"],
		    ["language_model.layers.22", "mlp.up_proj"],

		    ["language_model.layers.23", "mlp.down_proj"],
		    ["language_model.layers.23", "mlp.gate_proj"],
		    ["language_model.layers.23", "mlp.up_proj"],

		    ["language_model.layers.24", "mlp.down_proj"],
		    ["language_model.layers.24", "mlp.gate_proj"],
		    ["language_model.layers.24", "mlp.up_proj"],

		    ["language_model.layers.25", "mlp.down_proj"],
		    ["language_model.layers.25", "mlp.gate_proj"],
		    ["language_model.layers.25", "mlp.up_proj"],

		    ["language_model.layers.26", "mlp.down_proj"],
		    ["language_model.layers.26", "mlp.gate_proj"],
		    ["language_model.layers.26", "mlp.up_proj"],

		    ["language_model.layers.27", "mlp.down_proj"],
		    ["language_model.layers.27", "mlp.gate_proj"],
		    ["language_model.layers.27", "mlp.up_proj"],

		    ["language_model.layers.28", "mlp.down_proj"],
		    ["language_model.layers.28", "mlp.gate_proj"],
		    ["language_model.layers.28", "mlp.up_proj"],

		    ["language_model.layers.29", "mlp.down_proj"],
		    ["language_model.layers.29", "mlp.gate_proj"],
		    ["language_model.layers.29", "mlp.up_proj"],

		    ["language_model.layers.30", "mlp.down_proj"],
		    ["language_model.layers.30", "mlp.gate_proj"],
		    ["language_model.layers.30", "mlp.up_proj"],

		    ["language_model.layers.31", "mlp.down_proj"],
		    ["language_model.layers.31", "mlp.gate_proj"],
		    ["language_model.layers.31", "mlp.up_proj"],

		    ["language_model.layers.32", "mlp.down_proj"],
		    ["language_model.layers.32", "mlp.gate_proj"],
		    ["language_model.layers.32", "mlp.up_proj"],

		    ["language_model.layers.33", "mlp.down_proj"],
		    ["language_model.layers.33", "mlp.gate_proj"],
		    ["language_model.layers.33", "mlp.up_proj"],

		    ["language_model.layers.34", "mlp.down_proj"],
		    ["language_model.layers.34", "mlp.gate_proj"],
		    ["language_model.layers.34", "mlp.up_proj"],

		    ["language_model.layers.35", "mlp.down_proj"],
		    ["language_model.layers.35", "mlp.gate_proj"],
		    ["language_model.layers.35", "mlp.up_proj"]
		]

		attn_criteria = [
		    # q_proj + o_proj for all layers
		    ["language_model.layers.0", "self_attn.q_proj"],
		    ["language_model.layers.0", "self_attn.o_proj"],
		    ["language_model.layers.1", "self_attn.q_proj"],
		    ["language_model.layers.1", "self_attn.o_proj"],
		    ["language_model.layers.2", "self_attn.q_proj"],
		    ["language_model.layers.2", "self_attn.o_proj"],
		    ["language_model.layers.3", "self_attn.q_proj"],
		    ["language_model.layers.3", "self_attn.o_proj"],
		    ["language_model.layers.4", "self_attn.q_proj"],
		    ["language_model.layers.4", "self_attn.o_proj"],
		    ["language_model.layers.5", "self_attn.q_proj"],
		    ["language_model.layers.5", "self_attn.o_proj"],
		    ["language_model.layers.6", "self_attn.q_proj"],
		    ["language_model.layers.6", "self_attn.o_proj"],
		    ["language_model.layers.7", "self_attn.q_proj"],
		    ["language_model.layers.7", "self_attn.o_proj"],
		    ["language_model.layers.8", "self_attn.q_proj"],
		    ["language_model.layers.8", "self_attn.o_proj"],
		    ["language_model.layers.9", "self_attn.q_proj"],
		    ["language_model.layers.9", "self_attn.o_proj"],
		    ["language_model.layers.10", "self_attn.q_proj"],
		    ["language_model.layers.10", "self_attn.o_proj"],
		    ["language_model.layers.11", "self_attn.q_proj"],
		    ["language_model.layers.11", "self_attn.o_proj"],
		    ["language_model.layers.12", "self_attn.q_proj"],
		    ["language_model.layers.12", "self_attn.o_proj"],
		    ["language_model.layers.13", "self_attn.q_proj"],
		    ["language_model.layers.13", "self_attn.o_proj"],
		    ["language_model.layers.14", "self_attn.q_proj"],
		    ["language_model.layers.14", "self_attn.o_proj"],
		    ["language_model.layers.15", "self_attn.q_proj"],
		    ["language_model.layers.15", "self_attn.o_proj"],
		    ["language_model.layers.16", "self_attn.q_proj"],
		    ["language_model.layers.16", "self_attn.o_proj"],
		    ["language_model.layers.17", "self_attn.q_proj"],
		    ["language_model.layers.17", "self_attn.o_proj"],
		    ["language_model.layers.18", "self_attn.q_proj"],
		    ["language_model.layers.18", "self_attn.o_proj"],
		    ["language_model.layers.19", "self_attn.q_proj"],
		    ["language_model.layers.19", "self_attn.o_proj"],
		    ["language_model.layers.20", "self_attn.q_proj"],
		    ["language_model.layers.20", "self_attn.o_proj"],
		    ["language_model.layers.21", "self_attn.q_proj"],
		    ["language_model.layers.21", "self_attn.o_proj"],
		    ["language_model.layers.22", "self_attn.q_proj"],
		    ["language_model.layers.22", "self_attn.o_proj"],
		    ["language_model.layers.23", "self_attn.q_proj"],
		    ["language_model.layers.23", "self_attn.o_proj"],
		    ["language_model.layers.24", "self_attn.q_proj"],
		    ["language_model.layers.24", "self_attn.o_proj"],
		    ["language_model.layers.25", "self_attn.q_proj"],
		    ["language_model.layers.25", "self_attn.o_proj"],
		    ["language_model.layers.26", "self_attn.q_proj"],
		    ["language_model.layers.26", "self_attn.o_proj"],
		    ["language_model.layers.27", "self_attn.q_proj"],
		    ["language_model.layers.27", "self_attn.o_proj"],
		    ["language_model.layers.28", "self_attn.q_proj"],
		    ["language_model.layers.28", "self_attn.o_proj"],
		    ["language_model.layers.29", "self_attn.q_proj"],
		    ["language_model.layers.29", "self_attn.o_proj"],
		    ["language_model.layers.30", "self_attn.q_proj"],
		    ["language_model.layers.30", "self_attn.o_proj"],
		    ["language_model.layers.31", "self_attn.q_proj"],
		    ["language_model.layers.31", "self_attn.o_proj"],
		    ["language_model.layers.32", "self_attn.q_proj"],
		    ["language_model.layers.32", "self_attn.o_proj"],
		    ["language_model.layers.33", "self_attn.q_proj"],
		    ["language_model.layers.33", "self_attn.o_proj"],
		    ["language_model.layers.34", "self_attn.q_proj"],
		    ["language_model.layers.34", "self_attn.o_proj"],
		    ["language_model.layers.35", "self_attn.q_proj"],
		    ["language_model.layers.35", "self_attn.o_proj"],

		    # preserved k_proj / v_proj only (final_rank < 128)
		    ["language_model.layers.1", "self_attn.k_proj"],
		    ["language_model.layers.2", "self_attn.k_proj"],
		    ["language_model.layers.2", "self_attn.v_proj"],
		    ["language_model.layers.3", "self_attn.k_proj"],
		    ["language_model.layers.4", "self_attn.k_proj"],
		    ["language_model.layers.6", "self_attn.k_proj"],
		    ["language_model.layers.7", "self_attn.k_proj"],
		    ["language_model.layers.9", "self_attn.k_proj"],
		    ["language_model.layers.10", "self_attn.v_proj"],
		    ["language_model.layers.11", "self_attn.k_proj"],
		    ["language_model.layers.11", "self_attn.v_proj"],
		    ["language_model.layers.12", "self_attn.k_proj"],
		    ["language_model.layers.12", "self_attn.v_proj"],
		    ["language_model.layers.13", "self_attn.k_proj"],
		    ["language_model.layers.15", "self_attn.v_proj"],
		    ["language_model.layers.16", "self_attn.k_proj"],
		    ["language_model.layers.17", "self_attn.v_proj"],
		    ["language_model.layers.20", "self_attn.k_proj"],
		    ["language_model.layers.20", "self_attn.v_proj"],
		    ["language_model.layers.22", "self_attn.v_proj"],
		    ["language_model.layers.23", "self_attn.v_proj"],
		    ["language_model.layers.24", "self_attn.v_proj"],
		    ["language_model.layers.25", "self_attn.v_proj"],
		    ["language_model.layers.27", "self_attn.k_proj"],
		    ["language_model.layers.28", "self_attn.v_proj"],
		    ["language_model.layers.29", "self_attn.v_proj"],
		    ["language_model.layers.31", "self_attn.v_proj"],
		    ["language_model.layers.33", "self_attn.k_proj"],
		    ["language_model.layers.35", "self_attn.k_proj"]
		]
		
		mgr.init_vcon(verbose=True, criteria=mlp_criteria + attn_criteria)
		mgr.set_vcon_beta(beta=vcon_beta, criteria=mlp_criteria + attn_criteria)
	
	mgr.apply(verbose=True)

	
	'''
    mgr = managerClass(model)
    mgr.set("lrd", "rank", 128, [
        ["language_model", "mlp.down_proj"],
        ["language_model", "mlp.up_proj"],
        ["language_model", "mlp.gate_proj"]
    ], verbose=True)

    if use_vcon:
        mgr.init_vcon(verbose=True, criteria=[["language_model", "mlp.down_proj"],
        ["language_model", "mlp.up_proj"],
        ["language_model", "mlp.gate_proj"]])
        mgr.set_vcon_beta(beta=vcon_beta, criteria=[["language_model", "mlp.down_proj"],
        ["language_model", "mlp.up_proj"],
        ["language_model", "mlp.gate_proj"]])

    mgr.apply(verbose=True)
	'''
    
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

	if data_args.data_packing:
		data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
	else:
		data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
	if use_vcon:
		nof_samples = len(data_module["train_dataset"])
		batch_size = training_args.per_device_train_batch_size
		nof_steps_in_epoch = ceil(nof_samples / batch_size)
		grad_acc_steps = training_args.gradient_accumulation_steps
		nof_grad_acc_in_epoch = ceil(nof_steps_in_epoch / grad_acc_steps)
		vcon_grad_acc = nof_grad_acc_in_epoch * vcon_epochs
        
		VCON_beta_upd_callback = VCON_beta_upd(mgr, grad_acc_steps=grad_acc_steps, vcon_grad_acc=vcon_grad_acc, beta=vcon_beta)
		VCON_cancel_callback = VCON_cancel(mgr)
        
		trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module, callbacks=[VCON_beta_upd_callback, VCON_cancel_callback]
        )
	else:
        
		trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )
    
	'''
	if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
		logging.info("checkpoint found, resume training")
		trainer.train(resume_from_checkpoint=True)
	else:
		trainer.train()
    '''
	trainer.save_state()
	
    
	data_args.image_processor.save_pretrained(training_args.output_dir)

	model.config.use_cache = True

	safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
