"""
evaluate.py —— 计算分类指标：Overall Accuracy、Macro-F1、per-class F1、混淆矩阵。
仅依赖 sklearn，无内部模块依赖，可独立验证。
"""

import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
_AMP_ENABLED = torch.cuda.is_available()

import config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = config.CLASS_NAMES,
) -> dict:
    """
    计算所有分类指标。

    Parameters
    ----------
    y_true : (N,) int  真实标签
    y_pred : (N,) int  预测标签
    class_names : list[str]

    Returns
    -------
    dict with keys:
        overall_accuracy          : float
        macro_f1                  : float
        per_class_f1              : dict {class_name: f1}
        confusion_matrix          : np.ndarray (C, C)  原始计数
        normalized_confusion_matrix : np.ndarray (C, C)  按行归一化（真实类别为行）
    """
    oa       = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls  = f1_score(y_true, y_pred, average=None, zero_division=0)  # (C,)

    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm.astype(float) / row_sum, 0.0)

    return {
        "overall_accuracy":           float(oa),
        "macro_f1":                   float(macro_f1),
        "per_class_f1":               {name: float(per_cls[i]) for i, name in enumerate(class_names)},
        "confusion_matrix":           cm,
        "normalized_confusion_matrix": cm_norm,
    }


def evaluate_on_test(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    class_names: list = config.CLASS_NAMES,
) -> dict:
    """
    在测试集上完整推断，返回所有指标 + 额外信息（推断速度、参数量）。

    Returns
    -------
    dict: compute_metrics 的结果 +
        params_M              : float  模型参数量（百万）
        inference_ms_per_img  : float  每张图推断时间（毫秒）
    """
    model.eval()
    all_preds, all_labels = [], []

    # ── 计时 ──
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=_DEVICE_TYPE, enabled=_AMP_ENABLED):
                logits = model(imgs)

            preds = logits.argmax(dim=1)
            all_preds .extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_samples = len(all_labels)
    ms_per_img = elapsed * 1000 / n_samples

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        class_names,
    )

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    metrics["params_M"]             = round(params_m, 2)
    metrics["inference_ms_per_img"] = round(ms_per_img, 3)

    print(f"[evaluate] 测试集结果：")
    print(f"           Overall Accuracy = {metrics['overall_accuracy']:.4f}")
    print(f"           Macro-F1         = {metrics['macro_f1']:.4f}")
    print(f"           参数量           = {params_m:.1f}M")
    print(f"           推断速度         = {ms_per_img:.2f} ms/img")
    print(f"           Per-class F1:")
    for name, f1 in metrics["per_class_f1"].items():
        print(f"             {name:<25s} {f1:.4f}")

    return metrics
