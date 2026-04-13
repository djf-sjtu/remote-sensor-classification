"""
visualize.py —— 训练曲线、混淆矩阵热力图、两模型对比图。
所有图表保存为 .png，使用非交互式后端（适合无 GUI 服务器环境）。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 非交互式后端，服务器/无屏幕环境兼容
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config

# ── 全局字体设置（支持中文，若无中文字体则退回英文）──────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi":      150,
})

MODEL_DISPLAY = {
    "resnet50":  "ResNet-50",
    "vit_small": "ViT-Small/16",
}


# ─── 1. 训练曲线 ──────────────────────────────────────────────────────────────

def plot_training_curves(history: list, title: str, save_path: str):
    """
    绘制 2×2 训练曲线图：
    [0,0] Train Loss    [0,1] Val Loss
    [1,0] Train Acc     [1,1] Val Macro-F1

    Parameters
    ----------
    history   : list[dict]  每个 epoch 的指标字典（来自 train.run_training）
    title     : str         图表标题（含模型名和 seed）
    save_path : str         输出 .png 路径
    """
    epochs      = [r["epoch"]        for r in history]
    train_loss  = [r["train_loss"]   for r in history]
    val_loss    = [r["val_loss"]     for r in history]
    train_acc   = [r["train_acc"]    for r in history]
    val_acc     = [r["val_acc"]      for r in history]
    val_f1      = [r["val_macro_f1"] for r in history]

    best_ep = epochs[int(np.argmax(val_f1))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Loss ──
    for ax, (y1, y2, ylabel) in [
        (axes[0, 0], (train_loss, None,     "Loss")),
        (axes[0, 1], (val_loss,   None,     "Loss")),
    ]:
        pass  # 下方统一绘制

    def _plot(ax, ys, labels, colors, ylabel, mark_best=True):
        for y, lbl, col in zip(ys, labels, colors):
            ax.plot(epochs, y, color=col, label=lbl, linewidth=1.5)
        if mark_best:
            ax.axvline(x=best_ep, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                       label=f"best (ep={best_ep})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], [train_loss, val_loss], ["Train", "Val"],
          ["steelblue", "coral"],  "Loss")
    _plot(axes[0, 1], [train_acc,  val_acc],  ["Train Acc", "Val Acc"],
          ["steelblue", "coral"],  "Accuracy")
    _plot(axes[1, 0], [val_f1],               ["Val Macro-F1"],
          ["darkorange"],          "Macro-F1")
    # 学习率曲线
    if "lr" in history[0]:
        lrs = [float(r["lr"]) for r in history]
        axes[1, 1].plot(epochs, lrs, color="purple", linewidth=1.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title("LR Schedule")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 训练曲线已保存：{save_path}")


# ─── 2. 混淆矩阵热力图 ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm_norm: np.ndarray,
    class_names: list,
    title: str,
    save_path: str,
):
    """
    绘制归一化混淆矩阵热力图（按行归一化，行=真实类别）。

    Parameters
    ----------
    cm_norm     : (C, C) float  归一化混淆矩阵
    class_names : list[str]
    title       : str
    save_path   : str
    """
    try:
        import seaborn as sns
        _HAS_SEABORN = True
    except ImportError:
        _HAS_SEABORN = False

    C = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    if _HAS_SEABORN:
        sns.heatmap(
            cm_norm,
            annot=True, fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="lightgray",
            ax=ax,
            vmin=0, vmax=1,
        )
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 混淆矩阵已保存：{save_path}")


# ─── 3. 模型对比图 ────────────────────────────────────────────────────────────

def plot_comparison(all_results: list, save_path_prefix: str):
    """
    绘制两模型对比图（三张）：
    图1：OA & Macro-F1 分组柱状图（误差棒 = std）
    图2：Per-class F1 水平条形图对比
    图3：训练曲线均值对比（若 history 可用）

    Parameters
    ----------
    all_results       : list[dict]  每次实验的结果字典，含 model_name, seed,
                        overall_accuracy, macro_f1, per_class_f1 等字段
    save_path_prefix  : str  图表保存路径前缀，会自动追加 _fig1.png 等
    """
    os.makedirs(os.path.dirname(save_path_prefix) if os.path.dirname(save_path_prefix) else ".", exist_ok=True)

    models = config.MODELS  # ["resnet50", "vit_small"]

    # ── 整理统计数据 ──
    stats = {}
    for mn in models:
        runs = [r for r in all_results if r["model_name"] == mn]
        oas  = [r["overall_accuracy"] for r in runs]
        f1s  = [r["macro_f1"]         for r in runs]
        stats[mn] = {
            "oa_mean": np.mean(oas),  "oa_std": np.std(oas),
            "f1_mean": np.mean(f1s),  "f1_std": np.std(f1s),
        }

    # ── 图1：OA & Macro-F1 分组柱状图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x       = np.arange(2)          # OA, Macro-F1
    width   = 0.3
    colors  = ["steelblue", "coral"]

    for i, mn in enumerate(models):
        s    = stats[mn]
        vals = [s["oa_mean"],  s["f1_mean"]]
        errs = [s["oa_std"],   s["f1_std"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=MODEL_DISPLAY[mn], color=colors[i],
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Overall Accuracy", "Macro-F1"], fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.set_title("Model Comparison (mean ± std, 3 runs)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, save_path_prefix + "_metrics.png")

    # ── 图2：Per-class F1 水平条形图 ──
    cls_names = config.CLASS_NAMES
    cls_f1 = {}
    for mn in models:
        runs = [r for r in all_results if r["model_name"] == mn]
        # 每类 F1 取 3 次均值
        cls_f1[mn] = {
            name: np.mean([r["per_class_f1"][name] for r in runs])
            for name in cls_names
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    y     = np.arange(len(cls_names))
    height = 0.35

    for i, mn in enumerate(models):
        vals = [cls_f1[mn][name] for name in cls_names]
        ax.barh(y + (i - 0.5) * height, vals, height,
                label=MODEL_DISPLAY[mn], color=colors[i], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(cls_names, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.set_title("Per-class F1 Comparison (mean of 3 runs)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    _save(fig, save_path_prefix + "_per_class_f1.png")

    print(f"[visualize] 对比图已保存：{save_path_prefix}_metrics.png / _per_class_f1.png")


def _save(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
