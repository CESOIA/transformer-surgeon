import os
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# transformersurgeon (your package)
from transformersurgeon import (
    DistilBertForSequenceClassificationCompress,
    DistilBertCompressionSchemesManager,
)

VERBOSE = True
USE_GPU = True
BATCH_SIZE = 128


def build_dataloader(task: str, tokenizer, batch_size: int):
    task = task.lower()
    if task not in {"mnli", "qnli"}:
        raise ValueError("task must be one of: mnli, qnli")

    ds = load_dataset("glue", task)

    if task == "mnli":
        split = "validation_matched"
        sent1_key, sent2_key = "premise", "hypothesis"
    else:
        split = "validation"
        sent1_key, sent2_key = "question", "sentence"

    def preprocess(batch):
        enc = tokenizer(
            batch[sent1_key],
            batch[sent2_key],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        enc["labels"] = batch["label"]
        return enc

    ds_split = ds[split].map(preprocess, batched=True, remove_columns=ds[split].column_names)
    ds_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(ds_split, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    for batch in tqdm.tqdm(dataloader, desc="Evaluating accuracy..."):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mnli", "qnli"], default="qnli")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--compressed", action="store_true", help="Use transformersurgeon compressed model class")
    parser.add_argument("--do_compression", action="store_true", help="Apply compression via manager")
    parser.add_argument("--pruning_ratio", type=float, default=0.9)
    parser.add_argument("--pruning_mode", type=str, default="unstructured")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    num_labels = 3 if args.task == "mnli" else 2

    if args.compressed:
        model = DistilBertForSequenceClassificationCompress.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    if args.do_compression:
        if not args.compressed:
            raise RuntimeError("Compression manager expects a *Compress model class. Use --compressed.")

        manager = DistilBertCompressionSchemesManager(model)

        manager.set_pruning_ratio_all(args.pruning_ratio, verbose=VERBOSE)
        manager.set_pruning_mode_all(args.pruning_mode, verbose=VERBOSE)

        manager.apply_all(hard=False, verbose=VERBOSE)
        manager.update_config()

    dataloader = build_dataloader(args.task, tokenizer, args.batch_size)
    acc = evaluate(model, dataloader, device)
    print(f"Accuracy on GLUE-{args.task.upper()} validation: {acc:.2f}%")


if __name__ == "__main__":
    main()
