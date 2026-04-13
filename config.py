"""
config.py —— 所有超参数的唯一入口，无外部依赖。
修改实验参数时只需改这一个文件。
"""

import os

# ─── 路径 ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

STATS_CACHE  = os.path.join(OUTPUT_DIR, "stats_cache.npz")
SPLITS_CACHE = os.path.join(OUTPUT_DIR, "splits_cache.npz")
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR   = os.path.join(OUTPUT_DIR, "results")
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")

# ─── 数据 ─────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 10
NUM_BANDS    = 13
IMAGE_SIZE   = 224       # 原始 64×64 上采样到 224×224（两模型统一）
SPLIT_SEED   = 42        # 数据集划分种子，与训练种子解耦，固定不变
SPLIT_RATIO  = (0.70, 0.10, 0.20)   # train / val / test

# EuroSAT 类别名（按字母序，与 os.listdir 排序一致）
CLASS_NAMES = [
    "AnnualCrop",           # 0
    "Forest",               # 1
    "HerbaceousVegetation", # 2
    "Highway",              # 3
    "Industrial",           # 4
    "Pasture",              # 5
    "PermanentCrop",        # 6
    "Residential",          # 7
    "River",                # 8
    "SeaLake",              # 9
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Sentinel-2 波段到 ImageNet RGB 通道的物理对应关系
# key   = 目标通道索引（0~12，对应 B1~B12+B8A）
# value = ('copy', src_rgb_idx)  ← 直接复制对应 RGB 权重
#       = ('mean', None)         ← 用 RGB 三通道均值填充
# ImageNet RGB 索引约定：0=Red, 1=Green, 2=Blue
BAND_INIT_MAP = {
    0:  ('mean', None),  # B1  海岸气溶胶 443nm  — 无 RGB 对应
    1:  ('copy', 2),     # B2  蓝色      490nm  → Blue[2]
    2:  ('copy', 1),     # B3  绿色      560nm  → Green[1]
    3:  ('copy', 0),     # B4  红色      665nm  → Red[0]
    4:  ('copy', 0),     # B5  植被红边  705nm  ≈ Red（最近邻）
    5:  ('copy', 0),     # B6  植被红边  740nm  ≈ Red
    6:  ('mean', None),  # B7  植被红边  783nm  — NIR 边缘
    7:  ('mean', None),  # B8  近红外    842nm  — NIR
    8:  ('mean', None),  # B8A 窄带 NIR 865nm  — NIR
    9:  ('mean', None),  # B9  水汽      945nm  — 无对应
    10: ('mean', None),  # B10 卷云     1375nm  — 无对应
    11: ('mean', None),  # B11 SWIR    1610nm  — 无对应
    12: ('mean', None),  # B12 SWIR    2190nm  — 无对应
}

# ─── 训练（两模型严格共享，保证公平对比）─────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 64
NUM_WORKERS    = 4
LR             = 1e-4
WEIGHT_DECAY   = 0.05
LR_MIN         = 1e-6       # CosineAnnealingLR eta_min
GRAD_CLIP_NORM = 1.0        # 梯度裁剪 max_norm
MIXED_PRECISION = True      # torch.cuda.amp

# ─── 实验 ─────────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2]              # 3 次重复实验的随机种子
MODELS       = ["resnet50", "vit_small"]
BEST_METRIC  = "macro_f1"            # 以验证集 Macro-F1 保存 best checkpoint

MODEL_DISPLAY = {
    "resnet50":  "ResNet-50",
    "vit_small": "ViT-Small/16",
}
