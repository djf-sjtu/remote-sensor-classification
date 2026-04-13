"""
dataset.py —— 数据全流程：读取 .tif、分层划分、计算归一化统计量、构建 DataLoader。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import config

# ─── 工具：tif 读取 ────────────────────────────────────────────────────────────

def _read_tif(path: str) -> np.ndarray:
    """
    用 rasterio 读取 .tif，返回 float32 数组 (13, H, W)，已除以 10000 转为反射率。
    """
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32) / 10000.0  # uint16 → [0, ~1]
    return data


# ─── 1. 分层划分 ───────────────────────────────────────────────────────────────

def make_splits(
    data_root: str = config.DATA_ROOT,
    split_ratio: tuple = config.SPLIT_RATIO,
    split_seed: int = config.SPLIT_SEED,
    cache_path: str = config.SPLITS_CACHE,
):
    """
    扫描 data_root，构建文件路径和标签列表，执行两阶段分层划分。

    Returns
    -------
    all_paths  : list[str]
    all_labels : list[int]
    train_idx  : np.ndarray
    val_idx    : np.ndarray
    test_idx   : np.ndarray
    """
    # ── 尝试加载缓存 ──
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        if int(cache["split_seed"]) == split_seed:
            all_paths  = cache["all_paths"].tolist()
            all_labels = cache["all_labels"].tolist()
            train_idx  = cache["train_idx"]
            val_idx    = cache["val_idx"]
            test_idx   = cache["test_idx"]
            print(f"[dataset] 加载划分缓存：{cache_path}")
            print(f"          train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
            return all_paths, all_labels, train_idx, val_idx, test_idx

    # ── 扫描目录 ──
    all_paths, all_labels = [], []
    for class_name in sorted(os.listdir(data_root)):          # 字母序，与 CLASS_NAMES 一致
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = config.CLASS_TO_IDX[class_name]
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".tif"):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label)

    all_labels_arr = np.array(all_labels)
    indices = np.arange(len(all_paths))

    # ── 两阶段分层划分 ──
    # Step 1: 切出 test (20%)
    test_ratio = split_ratio[2]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=split_seed)
    temp_idx, test_idx = next(sss1.split(indices, all_labels_arr))

    # Step 2: 从 temp 中切出 val (10% / 80% = 12.5%)
    val_ratio_of_temp = split_ratio[1] / (split_ratio[0] + split_ratio[1])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_temp, random_state=split_seed)
    train_idx, val_idx = next(sss2.split(temp_idx, all_labels_arr[temp_idx]))
    train_idx = temp_idx[train_idx]
    val_idx   = temp_idx[val_idx]

    print(f"[dataset] 数据集划分完成（seed={split_seed}）")
    print(f"          总计={len(all_paths)}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ── 保存缓存 ──
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            split_seed  = split_seed,
            all_paths   = np.array(all_paths),
            all_labels  = np.array(all_labels),
            train_idx   = train_idx,
            val_idx     = val_idx,
            test_idx    = test_idx,
        )
        print(f"[dataset] 划分缓存已保存：{cache_path}")

    return all_paths, all_labels, train_idx, val_idx, test_idx


# ─── 2. 训练集统计量 ────────────────────────────────────────────────────────────

def compute_stats(
    all_paths: list,
    all_labels: list,
    train_idx: np.ndarray,
    cache_path: str = config.STATS_CACHE,
):
    """
    在训练集上用 Welford 在线算法计算 per-channel 均值和标准差。
    在原始 64×64 像素上统计（不做 Resize），避免插值引入误差。

    Returns
    -------
    mean : np.ndarray (13,)
    std  : np.ndarray (13,)
    """
    # ── 尝试加载缓存 ──
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path)
        print(f"[dataset] 加载统计量缓存：{cache_path}")
        print(f"          mean={cache['mean'].round(4)}")
        print(f"          std ={cache['std'].round(4)}")
        return cache["mean"], cache["std"]

    print(f"[dataset] 开始计算训练集统计量（{len(train_idx)} 张图像）…")

    # Welford 在线算法（单遍，内存高效）
    n       = 0
    mean    = np.zeros(config.NUM_BANDS, dtype=np.float64)
    M2      = np.zeros(config.NUM_BANDS, dtype=np.float64)

    for i, idx in enumerate(train_idx):
        img = _read_tif(all_paths[idx])          # (13, 64, 64) float32
        pixels = img.reshape(config.NUM_BANDS, -1)  # (13, 4096)

        # 对当前图像的每个通道像素逐一更新 Welford 累积量
        # 批量版：对每张图做批量 Welford 更新（图像为一个 batch）
        batch_n    = pixels.shape[1]             # = 64×64 = 4096
        batch_mean = pixels.mean(axis=1)         # (13,)
        batch_var  = pixels.var(axis=1)          # (13,)

        # 合并两组统计量（parallel Welford）
        new_n   = n + batch_n
        delta   = batch_mean - mean
        mean    = mean + delta * batch_n / new_n
        M2      = M2 + batch_var * batch_n + delta ** 2 * n * batch_n / new_n
        n       = new_n

        if (i + 1) % 2000 == 0:
            print(f"          [{i+1}/{len(train_idx)}]")

    std = np.sqrt(M2 / n)
    # 防止某通道方差为 0（极罕见，作保护）
    std = np.where(std < 1e-6, 1e-6, std)

    mean = mean.astype(np.float32)
    std  = std.astype(np.float32)

    print(f"[dataset] 统计量计算完成")
    print(f"          mean={mean.round(4)}")
    print(f"          std ={std.round(4)}")

    # ── 保存缓存 ──
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, mean=mean, std=std)
        print(f"[dataset] 统计量缓存已保存：{cache_path}")

    return mean, std


# ─── 3. Dataset ────────────────────────────────────────────────────────────────

class EuroSATDataset(Dataset):
    """
    读取 EuroSATallBands .tif 文件，返回归一化后的 (13, 224, 224) 张量。

    流程：rasterio 读取 → ÷10000 → ToTensor → Resize(224) → 空间增强（仅 train）→ 归一化
    """

    def __init__(
        self,
        paths: list,
        labels: list,
        mean: np.ndarray,
        std: np.ndarray,
        is_train: bool = False,
    ):
        self.paths    = paths
        self.labels   = labels
        self.mean     = torch.tensor(mean, dtype=torch.float32).view(config.NUM_BANDS, 1, 1)
        self.std      = torch.tensor(std,  dtype=torch.float32).view(config.NUM_BANDS, 1, 1)
        self.is_train = is_train

        # 按需导入 torchvision transforms（v2 支持任意通道 Tensor）
        try:
            import torchvision.transforms.v2 as T
        except ImportError:
            import torchvision.transforms as T

        self.resize = T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                               interpolation=T.InterpolationMode.BILINEAR,
                               antialias=True)

        if is_train:
            self.aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # 仅 0/90/180/270°，保持像素对齐，遥感图像无方向偏好
                T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5),
                T.RandomResizedCrop(
                    size=config.IMAGE_SIZE,
                    scale=(0.75, 1.0),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1. 读取 .tif → (13, 64, 64) float32，值域 [0, ~1]
        img = _read_tif(self.paths[idx])          # np.ndarray (13, 64, 64)
        img = torch.from_numpy(img)               # Tensor (13, 64, 64)

        # 2. Resize 到 224×224（两模型统一输入）
        img = self.resize(img)                    # (13, 224, 224)

        # 3. 空间增强（仅训练集）
        if self.aug is not None:
            img = self.aug(img)

        # 4. 归一化：(x - mean) / std
        img = (img - self.mean) / self.std

        label = self.labels[idx]
        return img, label


# ─── 4. DataLoader 工厂 ────────────────────────────────────────────────────────

def build_dataloaders(
    all_paths: list,
    all_labels: list,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Returns
    -------
    train_loader, val_loader, test_loader
    """
    def _subset(idx, is_train):
        paths  = [all_paths[i]  for i in idx]
        labels = [all_labels[i] for i in idx]
        return EuroSATDataset(paths, labels, mean, std, is_train=is_train)

    train_ds = _subset(train_idx, is_train=True)
    val_ds   = _subset(val_idx,   is_train=False)
    test_ds  = _subset(test_idx,  is_train=False)

    loader_kwargs = dict(
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        drop_last   = True,      # 避免末尾小 batch 影响 BatchNorm
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.BATCH_SIZE * 2,   # 验证时无梯度，batch 可更大
        shuffle     = False,
        drop_last   = False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = config.BATCH_SIZE * 2,
        shuffle     = False,
        drop_last   = False,
        **loader_kwargs,
    )

    print(f"[dataset] DataLoader 构建完成")
    print(f"          train batches={len(train_loader)}  "
          f"val batches={len(val_loader)}  "
          f"test batches={len(test_loader)}")

    return train_loader, val_loader, test_loader
