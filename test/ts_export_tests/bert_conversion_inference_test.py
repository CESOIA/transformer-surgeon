import torch

from transformers import AutoTokenizer

from transformersurgeon import BertForSequenceClassificationCompress
from transformersurgeon.utils import convert_for_export


MODEL_NAME = "bert-base-uncased"


def run_test(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = BertForSequenceClassificationCompress.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    converted_models = convert_for_export(
        model,
        options={"use_sdpa": True},
        verbose=True,
    )
    encoder = converted_models["bert"].to(device)
    encoder.eval()

    batch = tokenizer(
        "This is a compact test sentence for encoder export validation.",
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        embeddings = model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        hf_hidden = model.bert.encoder(embeddings, attention_mask=None).last_hidden_state
        ts_hidden = encoder(embeddings)

        hf_pooled = model.bert.pooler(hf_hidden)
        ts_pooled = model.bert.pooler(ts_hidden)

        hf_logits = model.classifier(model.dropout(hf_pooled))
        ts_logits = model.classifier(model.dropout(ts_pooled))

    hidden_diff = (hf_hidden - ts_hidden).abs()
    logits_diff = (hf_logits - ts_logits).abs()

    print("BERT hidden max abs diff:", float(hidden_diff.max().item()))
    print("BERT hidden mean abs diff:", float(hidden_diff.mean().item()))
    print("BERT logits max abs diff:", float(logits_diff.max().item()))
    print("BERT logits mean abs diff:", float(logits_diff.mean().item()))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    run_test(device)
