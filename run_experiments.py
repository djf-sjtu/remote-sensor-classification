"""
run_experiments.py —— 主入口。
功能：
  1. 计算/加载训练集统计量和数据集划分
  2. 遍历 模型 × 种子，执行完整实验（支持断点续跑）
  3. 聚合 3 次实验的均值 ± 标准差，输出 all_runs.csv 和 summary.csv
  4. 生成对比图表

用法：
  # 完整运行（6 次实验）
  python run_experiments.py

  # 仅跑指定模型和种子（调试）
  python run_experiments.py --model resnet50 --seed 0 --epochs 3

  # 重新生成对比图（不重新训练）
  python run_experiments.py --plot_only
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

import config
from dataset  import make_splits, compute_stats, build_dataloaders
from models   import build_model
from train    import run_training, load_best_checkpoint
from evaluate import evaluate_on_test
from visualize import (plot_training_curves, plot_confusion_matrix,
                        plot_comparison)


# ─── 随机种子固定 ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """固定 Python / NumPy / PyTorch / CUDA 的随机状态，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── 单次实验 ─────────────────────────────────────────────────────────────────

def run_single_experiment(
    model_name: str,
    seed: int,
    all_paths: list,
    all_labels: list,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    mean: np.ndarray,
    std:  np.ndarray,
    device: torch.device,
    epochs_override: int = None,
) -> dict:
    """
    一次完整实验流程：
      set_seed → DataLoader → build_model → train → load best → test eval → 出图

    Returns
    -------
    result : dict  含所有测试指标及元信息
    """
    print(f"\n{'='*60}")
    print(f"  实验：{model_name}  seed={seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # ── 路径规划 ──
    tag      = f"{model_name}_seed{seed}"
    ckpt_path    = os.path.join(config.CKPT_DIR,   f"{tag}_best.pth")
    log_csv_path = os.path.join(config.RESULT_DIR, f"{tag}_epoch_log.csv")
    curve_path   = os.path.join(config.FIG_DIR,    f"{tag}_curves.png")
    cm_path      = os.path.join(config.FIG_DIR,    f"{tag}_confusion.png")

    # ── DataLoader（不同 seed 对训练顺序有影响，通过 set_seed 保证）──
    train_loader, val_loader, test_loader = build_dataloaders(
        all_paths, all_labels, train_idx, val_idx, test_idx, mean, std
    )

    # ── 构建模型 ──
    model = build_model(model_name).to(device)

    # ── 训练 ──
    epochs_bak = config.EPOCHS
    if epochs_override is not None:
        config.EPOCHS = epochs_override

    t0 = time.time()
    history, best_val_f1 = run_training(
        model, train_loader, val_loader, ckpt_path, log_csv_path
    )
    train_min = (time.time() - t0) / 60
    config.EPOCHS = epochs_bak   # 恢复

    # ── 绘制训练曲线 ──
    plot_training_curves(
        history,
        title     = f"{config.MODEL_DISPLAY.get(model_name, model_name)} | seed={seed}",
        save_path = curve_path,
    )

    # ── 加载 best checkpoint，测试集评估 ──
    load_best_checkpoint(model, ckpt_path, device)
    metrics = evaluate_on_test(model, test_loader, device)

    # ── 绘制混淆矩阵 ──
    plot_confusion_matrix(
        cm_norm     = metrics["normalized_confusion_matrix"],
        class_names = config.CLASS_NAMES,
        title       = f"{config.MODEL_DISPLAY.get(model_name, model_name)} | seed={seed} | "
                      f"Macro-F1={metrics['macro_f1']:.4f}",
        save_path   = cm_path,
    )

    # ── 整理结果 ──
    result = {
        "model_name":        model_name,
        "seed":              seed,
        "overall_accuracy":  metrics["overall_accuracy"],
        "macro_f1":          metrics["macro_f1"],
        "per_class_f1":      metrics["per_class_f1"],
        "params_M":          metrics["params_M"],
        "inference_ms_per_img": metrics["inference_ms_per_img"],
        "train_time_min":    round(train_min, 1),
        "best_val_f1":       round(best_val_f1, 4),
    }

    print(f"\n[run] {tag} 完成  OA={result['overall_accuracy']:.4f}  "
          f"Macro-F1={result['macro_f1']:.4f}  "
          f"训练时长={train_min:.1f}min")
    return result


# ─── 聚合与写 CSV ─────────────────────────────────────────────────────────────

def save_all_runs_csv(all_results: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model_name", "seed",
        "overall_accuracy", "macro_f1",
        "params_M", "inference_ms_per_img", "train_time_min",
    ] + [f"f1_{name}" for name in config.CLASS_NAMES]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "model_name":          r["model_name"],
                "seed":                r["seed"],
                "overall_accuracy":    round(r["overall_accuracy"], 4),
                "macro_f1":            round(r["macro_f1"], 4),
                "params_M":            r["params_M"],
                "inference_ms_per_img": r["inference_ms_per_img"],
                "train_time_min":      r["train_time_min"],
            }
            for name in config.CLASS_NAMES:
                row[f"f1_{name}"] = round(r["per_class_f1"][name], 4)
            writer.writerow(row)
    print(f"[run] all_runs.csv 已保存：{path}")


def save_summary_csv(all_results: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model_name",
        "oa_mean", "oa_std",
        "f1_mean", "f1_std",
        "params_M", "inference_ms_per_img",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mn in config.MODELS:
            runs = [r for r in all_results if r["model_name"] == mn]
            oas  = [r["overall_accuracy"] for r in runs]
            f1s  = [r["macro_f1"]         for r in runs]
            row  = {
                "model_name":            mn,
                "oa_mean":               round(np.mean(oas), 4),
                "oa_std":                round(np.std(oas),  4),
                "f1_mean":               round(np.mean(f1s), 4),
                "f1_std":                round(np.std(f1s),  4),
                "params_M":              runs[0]["params_M"],
                "inference_ms_per_img":  round(np.mean([r["inference_ms_per_img"] for r in runs]), 3),
            }
            writer.writerow(row)
            print(f"[run] {mn:12s}  OA={row['oa_mean']:.4f}±{row['oa_std']:.4f}  "
                  f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}")
    print(f"[run] summary.csv 已保存：{path}")


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EuroSAT ResNet-50 vs ViT-Small 对比实验")
    parser.add_argument("--model",     type=str, default=None,
                        choices=["resnet50", "vit_small"],
                        help="仅跑指定模型（不指定则跑全部）")
    parser.add_argument("--seed",      type=int, default=None,
                        help="仅跑指定 seed（不指定则跑全部）")
    parser.add_argument("--epochs",    type=int, default=None,
                        help="覆盖训练 epoch 数（调试用）")
    parser.add_argument("--plot_only", action="store_true",
                        help="仅重新生成对比图，不重新训练")
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] 使用设备：{device}")
    if device.type == "cuda":
        print(f"      GPU：{torch.cuda.get_device_name(0)}")

    # ── 数据准备（一次性，所有实验共享）──
    all_paths, all_labels, train_idx, val_idx, test_idx = make_splits()
    mean, std = compute_stats(all_paths, all_labels, train_idx)

    # ── 确定要跑的实验列表 ──
    models_to_run = [args.model] if args.model else config.MODELS
    seeds_to_run  = [args.seed]  if args.seed  is not None else config.SEEDS

    # ── 如果只出图，加载已有结果 ──
    if args.plot_only:
        all_results = _load_existing_results(models_to_run, seeds_to_run)
        if all_results:
            _generate_comparison(all_results)
        return

    # ── 遍历实验，支持断点续跑 ──
    all_results = []
    for mn in models_to_run:
        for seed in seeds_to_run:
            result_json = os.path.join(config.RESULT_DIR, f"{mn}_seed{seed}_result.json")

            # 断点续跑：已完成则直接加载
            if os.path.exists(result_json):
                with open(result_json, "r", encoding="utf-8") as f:
                    result = json.load(f)
                print(f"[run] 跳过已完成实验：{mn} seed={seed}  "
                      f"OA={result['overall_accuracy']:.4f}  F1={result['macro_f1']:.4f}")
                all_results.append(result)
                continue

            result = run_single_experiment(
                model_name      = mn,
                seed            = seed,
                all_paths       = all_paths,
                all_labels      = all_labels,
                train_idx       = train_idx,
                val_idx         = val_idx,
                test_idx        = test_idx,
                mean            = mean,
                std             = std,
                device          = device,
                epochs_override = args.epochs,
            )

            # 保存单次结果 JSON（用于断点续跑）
            os.makedirs(config.RESULT_DIR, exist_ok=True)
            with open(result_json, "w", encoding="utf-8") as f:
                # numpy int/float 不能直接 json 序列化，转换一下
                result_serializable = _to_serializable(result)
                json.dump(result_serializable, f, ensure_ascii=False, indent=2)

            all_results.append(result)

    # ── 汇总输出 ──
    if len(all_results) == len(config.MODELS) * len(config.SEEDS):
        # 全部实验完成时才输出对比图
        save_all_runs_csv(all_results, os.path.join(config.RESULT_DIR, "all_runs.csv"))
        save_summary_csv (all_results, os.path.join(config.RESULT_DIR, "summary.csv"))
        _generate_comparison(all_results)
    else:
        # 部分实验，只输出已有的 CSV
        save_all_runs_csv(all_results, os.path.join(config.RESULT_DIR, "all_runs_partial.csv"))
        print(f"[run] 当前完成 {len(all_results)} / {len(config.MODELS)*len(config.SEEDS)} 次实验")

    print("\n[run] 全部完成！输出目录：", config.OUTPUT_DIR)


def _generate_comparison(all_results):
    save_prefix = os.path.join(config.FIG_DIR, "comparison")
    os.makedirs(config.FIG_DIR, exist_ok=True)
    plot_comparison(all_results, save_prefix)


def _load_existing_results(models, seeds):
    results = []
    for mn in models:
        for seed in seeds:
            path = os.path.join(config.RESULT_DIR, f"{mn}_seed{seed}_result.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    results.append(json.load(f))
    return results


def _to_serializable(obj):
    """递归将 numpy 类型转为 Python 原生类型，以支持 json.dump。"""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─── 补充：MODEL_DISPLAY 在 visualize 中使用，config 里补充 ──────────────────
# （visualize.py 里直接引用 config.MODEL_DISPLAY，在此处补充到 config 模块）
if not hasattr(config, "MODEL_DISPLAY"):
    config.MODEL_DISPLAY = {
        "resnet50":  "ResNet-50",
        "vit_small": "ViT-Small/16",
    }


if __name__ == "__main__":
    main()
