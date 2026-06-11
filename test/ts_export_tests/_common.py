import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN_MODEL = "Qwen/Qwen2-0.5B-Instruct"
VIT_MODEL = "google/vit-base-patch16-224"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
    "<|im_start|>user\n{instruction}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)
TEST_PROMPT = "Tell me one short fact about France."

VIT_HIDDEN_ATOL = 1e-4
VIT_LOGIT_ATOL = 1e-3


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def logits_to_id(logits, temperature=0.0):
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
