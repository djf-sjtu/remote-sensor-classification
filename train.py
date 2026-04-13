"""
train.py —— 训练循环、验证循环、优化器/调度器工厂、checkpoint 保存。
两个模型使用完全相同的训练配置，保证公平对比。
"""

import os
import time
import csv
import torch
import torch.nn as nn

import config

# 兼容 CPU / GPU 的 autocast 和 GradScaler
_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
_AMP_ENABLED = config.MIXED_PRECISION and torch.cuda.is_available()


def _autocast():
    return torch.amp.autocast(device_type=_DEVICE_TYPE, enabled=_AMP_ENABLED)


def _make_scaler():
    return torch.amp.GradScaler(device=_DEVICE_TYPE, enabled=_AMP_ENABLED)


# ─── 优化器 & 调度器 ──────────────────────────────────────────────────────────

def build_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.LR,
        weight_decay = config.WEIGHT_DECAY,
        betas        = (0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = config.EPOCHS,
        eta_min = config.LR_MIN,
    )
    return optimizer, scheduler


# ─── 单 epoch 训练 ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with _autocast():
            logits = model(imgs)
            loss   = loss_fn(logits, labels)

        if torch.isnan(loss):
            print("[train] 警告：检测到 NaN loss，跳过此 batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += bs

    return {"loss": total_loss / total, "acc": correct / total}


# ─── 验证 ─────────────────────────────────────────────────────────────────────

def validate(model, loader, loss_fn, device):
    from evaluate import compute_metrics
    import numpy as np

    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with _autocast():
                logits = model(imgs)
                loss   = loss_fn(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)
            all_preds .extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), config.CLASS_NAMES)
    return {
        "loss":     total_loss / total,
        "acc":      metrics["overall_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


# ─── 完整训练流程 ─────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, ckpt_path, log_csv_path):
    device = next(model.parameters()).device

    loss_fn              = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_scheduler(model)
    scaler               = _make_scaler()

    os.makedirs(os.path.dirname(ckpt_path),    exist_ok=True)
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    fieldnames = ["epoch", "train_loss", "train_acc",
                  "val_loss", "val_acc", "val_macro_f1", "lr"]
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    history, best_val_f1 = [], -1.0
    t0 = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        t_ep          = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_metrics   = validate(model, val_loader, loss_fn, device)
        current_lr    = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_metrics["loss"],     4),
            "train_acc":    round(train_metrics["acc"],      4),
            "val_loss":     round(val_metrics["loss"],       4),
            "val_acc":      round(val_metrics["acc"],        4),
            "val_macro_f1": round(val_metrics["macro_f1"],  4),
            "lr":           f"{current_lr:.2e}",
        }
        writer.writerow(row)
        csv_file.flush()
        history.append(row)

        print(f"  Epoch {epoch:3d}/{config.EPOCHS} | "
              f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
              f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
              f"f1={val_metrics['macro_f1']:.4f} | "
              f"lr={current_lr:.2e} | {time.time()-t_ep:.0f}s")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "best_macro_f1":    best_val_f1,
            }, ckpt_path)
            print(f"  Best checkpoint saved (val macro_f1={best_val_f1:.4f})")

    print(f"[train] 训练完成，耗时 {(time.time()-t0)/60:.1f} 分钟，最优 val Macro-F1={best_val_f1:.4f}")
    csv_file.close()
    return history, best_val_f1


# ─── 加载 checkpoint ──────────────────────────────────────────────────────────

def load_best_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    best_f1 = ckpt.get("best_macro_f1", -1.0)
    print(f"[train] 已加载 checkpoint（epoch={ckpt['epoch']}, best_f1={best_f1:.4f}）：{ckpt_path}")
    return best_f1
