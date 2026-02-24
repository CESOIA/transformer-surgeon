#!/usr/bin/env python3
"""
Train BERT/DistilBERT with Transformersurgeon compression on GLUE (MNLI/QNLI).
"""

import os
import argparse
import random

import numpy as np
import torch
import tqdm
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from torch.amp import autocast, GradScaler

from transformersurgeon import (
    BertForSequenceClassificationCompress,
    BertCompressionSchemesManager,
    DistilBertForSequenceClassificationCompress,
    DistilBertCompressionSchemesManager,
)


# -------- fixed defaults (you can still override via CLI) --------
DEFAULT_BATCH_SIZE = 16
DEFAULT_VAL_BATCH_SIZE = 128
DEFAULT_EPOCHS = 8
DEFAULT_LR = 2e-5
DEFAULT_EARLYSTOP_PATIENCE = 5
# ---------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_run_dir(base: str) -> str:
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    take = 1
    while os.path.exists(f"{base}_take{take}"):
        take += 1
    run_dir = f"{base}_take{take}"
    os.makedirs(run_dir)
    return run_dir


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_glue_loaders(task: str, tokenizer, workers: int, batch_size: int, val_batch_size: int):
    task = task.lower()
    ds = load_dataset("glue", task)

    if task == "mnli":
        sent1_key, sent2_key = "premise", "hypothesis"
        train_split = ds["train"]
        val_splits = {
            "val_matched": ds["validation_matched"],
            "val_mismatched": ds["validation_mismatched"],
        }
        num_labels = 3
    elif task == "qnli":
        sent1_key, sent2_key = "question", "sentence"
        train_split = ds["train"]
        val_splits = {"val": ds["validation"]}
        num_labels = 2
    else:
        raise ValueError("task must be one of: mnli, qnli")

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

    train_tok = train_split.map(preprocess, batched=True, remove_columns=train_split.column_names)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    val_tok = {}
    for name, split in val_splits.items():
        tmp = split.map(preprocess, batched=True, remove_columns=split.column_names)
        tmp.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_tok[name] = tmp

    def dl_kwargs(bs, shuffle):
        kw = dict(
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        if workers > 0:
            kw.update(dict(persistent_workers=True, prefetch_factor=2))
        return kw

    train_loader = DataLoader(train_tok, **dl_kwargs(batch_size, shuffle=True))
    val_loaders = {name: DataLoader(split, **dl_kwargs(val_batch_size, shuffle=False)) for name, split in val_tok.items()}
    return train_loader, val_loaders, num_labels


@torch.no_grad()
def evaluate(model, manager, loader, device, use_amp: bool, amp_dtype: torch.dtype, criterion, apply_pruning: bool):
    model.eval()
    if apply_pruning:
        manager.apply_all()

    total = 0
    correct = 0
    cum_loss = 0.0

    for batch in tqdm.tqdm(loader, desc="Evaluating..."):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type=device.type, dtype=amp_dtype if device.type == "cuda" else None):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, labels)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        cum_loss += loss.item()

        del input_ids, attention_mask, labels, logits, loss

    if apply_pruning:
        manager.restore_all()

    acc = 100.0 * correct / max(total, 1)
    avg_loss = cum_loss / max(len(loader), 1)
    return acc, avg_loss


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train BERT/DistilBERT with compression on GLUE (MNLI/QNLI)")

    parser.add_argument("--arch", choices=["bert", "distilbert"], default="distilbert")
    parser.add_argument("--task", choices=["mnli", "qnli"], default="qnli")
    parser.add_argument("--model_name", type=str, default="run_glue_compress", help="Run directory name")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="HF checkpoint override (base model)")
    parser.add_argument("--load_model_path", type=str, default=None, help="Load a pretrained checkpoint path/dir")

    # infra
    parser.add_argument("--no_gpu", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", default=False)

    # training (NO warmup; early stopping only)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val_batch_size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--earlystop_patience", type=int, default=DEFAULT_EARLYSTOP_PATIENCE)
    parser.add_argument("--force_adam", action="store_true", default=False)

    # precision
    parser.add_argument("--datatype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--use_amp", action="store_true", default=False)

    # save
    parser.add_argument("--store_model", action="store_true", default=False)

    # compression knobs
    parser.add_argument("--pruning_ratio", type=float, default=0.0)
    parser.add_argument("--pruning_epochs", type=int, default=0,
                        help="epochs to reach target pruning ratio (ratio changes per iteration)")
    parser.add_argument("--pruning_type", type=str, default="unstructured", choices=["structured", "unstructured"])

    parser.add_argument("--lrd_rank", type=str, default="full",
                        help='LRD rank integer or "full" to disable')
    parser.add_argument("--lrd_epochs", type=int, default=0,
                        help="epochs to reach target LRD rank (rank changes per iteration)")

    parser.add_argument("--vcon_epochs", type=int, default=2)
    parser.add_argument("--freeze_original", action="store_true", default=False)

    args = parser.parse_args()

    # validate pruning
    if args.pruning_ratio < 0.0 or args.pruning_ratio >= 1.0:
        raise ValueError("Pruning ratio must be in the range [0.0, 1.0).")

    # dtypes
    if args.datatype == "float16":
        PARAM_DTYPE = torch.float16
        AMP_DTYPE = torch.float16
    elif args.datatype == "bfloat16":
        PARAM_DTYPE = torch.bfloat16
        AMP_DTYPE = torch.bfloat16
    else:
        PARAM_DTYPE = torch.float32
        AMP_DTYPE = torch.float16  # if amp is enabled on cuda

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")

    TARGET_LRD_RANK = int(args.lrd_rank) if args.lrd_rank != "full" else "full"
    assert not (args.pruning_ratio > 0.0 and TARGET_LRD_RANK != "full"), \
        "LRD and Pruning cannot be applied together in this implementation."

    # checkpoints
    if args.model_checkpoint is not None:
        base_ckpt = args.model_checkpoint
    else:
        base_ckpt = "bert-base-cased" if args.arch == "bert" else "distilbert-base-cased"
    pretrained_ckpt = args.load_model_path if args.load_model_path is not None else base_ckpt

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt, use_fast=True)
    train_loader, val_loaders, num_labels = build_glue_loaders(
        args.task, tokenizer, args.workers, args.batch_size, args.val_batch_size
    )

    # model + manager
    if args.arch == "bert":
        model = BertForSequenceClassificationCompress.from_pretrained(
            pretrained_ckpt,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            dtype=PARAM_DTYPE,
        )
        manager = BertCompressionSchemesManager(model)
    else:
        model = DistilBertForSequenceClassificationCompress.from_pretrained(
            pretrained_ckpt,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            dtype=PARAM_DTYPE,
        )
        manager = DistilBertCompressionSchemesManager(model)

    # init manager schemes
    if args.pruning_ratio > 0.0:
        manager.set_pruning_mode_all(args.pruning_type, verbose=args.verbose)

    if args.vcon_epochs > 0:
        manager.init_vcon_all(verbose=args.verbose)
        if args.freeze_original:
            manager.freeze_uncompressed_vcon_all(verbose=args.verbose)

    # optimizer
    optimclass = torch.optim.Adam if args.force_adam else torch.optim.AdamW
    optimizer = optimclass(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # AMP scaler
    use_scaler = args.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_scaler)

    # bookkeeping
    run_dir = make_run_dir(args.model_name)
    writer = tb.SummaryWriter(log_dir=run_dir)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    model.to(device)

    # vcon params
    vcon_beta = 1.0
    vcon_delta = 0.0 if args.vcon_epochs <= 0 else -1.0 / args.vcon_epochs
    vcon_enabled = args.vcon_epochs > 0

    pruning_ratio = 0.0
    pruning_delta = (
        args.pruning_ratio
        if args.pruning_epochs <= 0
        else args.pruning_ratio / (args.pruning_epochs * len(train_loader))
    )

    if TARGET_LRD_RANK == "full":
        current_lrd_rank = "full"
        old_lrd_rank = "full"
        lrd_rank_float = None
        lrd_delta = 0.0
    else:
        start_rank = model.config.hidden_size - 1
        lrd_rank_float = float(start_rank)
        delta_total = (TARGET_LRD_RANK - model.config.hidden_size + 1)
        lrd_delta = (
            delta_total
            if args.lrd_epochs <= 0
            else delta_total / (args.lrd_epochs * len(train_loader))
        )
        current_lrd_rank = start_rank
        old_lrd_rank = "full"

    earlystop = EarlyStopping(patience=args.earlystop_patience)
    primary_val_name = "val_matched" if args.task == "mnli" else "val"

    GRADIENT_ACCUMULATION_STEPS = 1

    # ---------------- TRAIN ----------------
    for epoch in range(args.epochs):
        
        model.train()
        cumulative_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        amp_str = f", amp scale {scaler.get_scale():.4e}" if use_scaler else ""
        for batch_idx, batch in enumerate(
            tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}{amp_str}")
        ):
            # Update pruning ratio
            if args.pruning_ratio > 0.0 and pruning_ratio < args.pruning_ratio:
                pruning_ratio = min(args.pruning_ratio, pruning_ratio + pruning_delta)
                manager.set_pruning_ratio_all(ratio=pruning_ratio, verbose=args.verbose)

            # Update LRD ranks
            if TARGET_LRD_RANK != "full" and current_lrd_rank > TARGET_LRD_RANK:
                lrd_rank_float += lrd_delta
                current_lrd_rank = int(round(lrd_rank_float))

            # Setup/update VCON beta at start of each epoch
            if vcon_enabled and batch_idx == 0:
                if vcon_beta > 0.0:
                    manager.set_vcon_beta_all(beta=vcon_beta, verbose=args.verbose)
                    vcon_beta += vcon_delta
                if vcon_beta <= 0.0:
                    manager.cancel_vcon_all(verbose=args.verbose)
                    vcon_enabled = False

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Apply pruning / compression before forward
            if args.pruning_ratio > 0.0:
                manager.apply_all()

            # Apply LRD when rank changes
            if TARGET_LRD_RANK != "full" and old_lrd_rank != current_lrd_rank:
                manager.restore_all(topology=True)
                manager.set_lrd_rank_all(rank=current_lrd_rank, verbose=args.verbose)
                manager.apply_all(verbose=args.verbose)
                # Reinitialize optimizer after param/topology change
                optimizer = optimclass(model.parameters(), lr=optimizer.param_groups[0]["lr"])
                old_lrd_rank = current_lrd_rank

            # Forward
            if use_scaler:
                with autocast(device_type=device.type, dtype=AMP_DTYPE):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = criterion(logits, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                if args.use_amp:
                    with autocast(device_type=device.type, dtype=AMP_DTYPE if device.type == "cuda" else None):
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        loss = criterion(logits, labels)
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = criterion(logits, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

            cumulative_loss += float(loss.item() * GRADIENT_ACCUMULATION_STEPS)

            # Backward
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.pruning_ratio > 0.0:
                manager.restore_all()

            # Early Stopping
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            del input_ids, attention_mask, labels, logits, loss
            if device.type == "cuda" and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        train_loss = cumulative_loss / max(len(train_loader), 1)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Pruning-Ratio", pruning_ratio if args.pruning_ratio > 0.0 else 0.0, epoch)
        writer.add_scalar("LRD-Rank", 0.0 if TARGET_LRD_RANK == "full" else float(current_lrd_rank), epoch)
        writer.add_scalar("VCON-Beta", vcon_beta if vcon_enabled else 0.0, epoch)
        writer.add_scalar("Learning-Rate", optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}")

        # ---------------- VALIDATION ----------------
        val_losses = {}
        for split_name, loader in val_loaders.items():
            acc, vloss = evaluate(
                model=model,
                manager=manager,
                loader=loader,
                device=device,
                use_amp=args.use_amp,
                amp_dtype=AMP_DTYPE,
                criterion=criterion,
                apply_pruning=(args.pruning_ratio > 0.0),
            )
            val_losses[split_name] = vloss
            writer.add_scalar(f"Val/{split_name}_Accuracy", acc, epoch)
            writer.add_scalar(f"Val/{split_name}_Loss", vloss, epoch)
            print(f"Epoch [{epoch+1}/{args.epochs}] {split_name}: loss={vloss:.4f}, acc={acc:.2f}%")

        # Early stopping on primary validation split loss
        primary_vloss = val_losses[primary_val_name]
        if earlystop.step(primary_vloss):
            print(f"EarlyStopping triggered (patience={args.earlystop_patience}). Stopping at epoch {epoch+1}.")
            break

    writer.close()

    if args.store_model:
        model.save_pretrained(run_dir)

    del train_loader
    for k in list(val_loaders.keys()):
        del val_loaders[k]


if __name__ == "__main__":
    main()
