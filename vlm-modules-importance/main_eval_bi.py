# main_eval_bi.py

import torch
from transformers import AutoProcessor, AutoTokenizer
from indexing.qwen_indexing import QWEN2_5_VL_IMP_INDEXING as INDEXING

from transformersurgeon import (
    Qwen2_5_VLForConditionalGenerationCompress,
)

from data.QwenVL_calibDataset import QwenVLCalibDataset, collate_fn_qwen
from importance_eval_methods.BI import eval_bi
from utils.utils import sort_bi_scores

json_path = "/ibex/user/antonic/CESOIA/datasets/COCO/annotations/train2017_qa_cats_chair2.jsonl"
image_root = "/ibex/user/antonic/CESOIA/datasets/COCO/train2017/"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Model + Processor
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGenerationCompress.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(device)

# Dataset
dataset = QwenVLCalibDataset(json_path, image_root)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn_qwen)

# === Compute BI ===
bi_scores = eval_bi(model, loader, INDEXING, processor)

# === Sort results ===
sorted_results = sort_bi_scores(bi_scores)

print(sorted_results["vision_block_sorted"])
print(sorted_results["text_block_sorted"])
print(sorted_results["vision_mlp_attn_sorted"])
print(sorted_results["text_mlp_attn_sorted"])
