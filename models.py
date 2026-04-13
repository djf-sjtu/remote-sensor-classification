"""
models.py —— ResNet-50 和 ViT-Small/16 的构建。
两个模型共享通道扩展逻辑（expand_first_layer），使用 ImageNet 预训练权重
并将输入通道从 3 扩展到 13（基于 Sentinel-2 波段物理对应关系）。
"""

import torch
import torch.nn as nn

import config


# ─── 核心工具：通道扩展 ────────────────────────────────────────────────────────

def expand_first_layer(
    original_weight: torch.Tensor,
    band_init_map: dict = config.BAND_INIT_MAP,
) -> torch.Tensor:
    """
    将预训练第一层卷积权重从 3 通道扩展到 13 通道。

    Parameters
    ----------
    original_weight : Tensor (out_c, 3, kH, kW)
        ImageNet 预训练的第一层卷积权重（RGB 3 通道）。
    band_init_map : dict
        Sentinel-2 各波段到 RGB 通道的映射，来自 config.BAND_INIT_MAP。

    Returns
    -------
    new_weight : Tensor (out_c, 13, kH, kW)

    初始化策略
    ----------
    - 'copy' 策略：直接复制对应 RGB 通道权重（B2→Blue, B3→Green, B4→Red 等）
    - 'mean' 策略：用 RGB 三通道均值填充（无物理对应的波段）
    - 整体乘以缩放因子 3/13：保持卷积输出的激活量级与 3 通道时一致
      （原3通道→13通道，加权和期望扩大约4.3倍，缩放回来）
    """
    out_c, _, kH, kW = original_weight.shape
    num_new = config.NUM_BANDS  # 13

    # RGB 三通道均值，形状 (out_c, kH, kW)
    rgb_mean = original_weight.mean(dim=1)

    new_weight = torch.zeros(out_c, num_new, kH, kW, dtype=original_weight.dtype)

    for ch_idx, (strategy, src_idx) in band_init_map.items():
        if strategy == 'copy':
            new_weight[:, ch_idx] = original_weight[:, src_idx]
        else:  # 'mean'
            new_weight[:, ch_idx] = rgb_mean

    # 缩放因子：保持激活量级不变
    scale = 3.0 / num_new
    new_weight = new_weight * scale

    return new_weight


# ─── ResNet-50 ────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    """
    构建 ResNet-50，将第一层卷积扩展为 13 通道，分类头改为 num_classes 输出。

    Steps
    -----
    1. 加载 ImageNet 预训练权重（torchvision）
    2. 保存 conv1.weight (64, 3, 7, 7)
    3. 替换 conv1 为 Conv2d(13, 64, 7, 7, stride=2, padding=3, bias=False)
    4. 用 expand_first_layer 初始化新 conv1 权重
    5. 替换 fc 为 Linear(2048, num_classes)
    """
    import torchvision.models as tvm

    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = tvm.resnet50(weights=weights)

    # ── 扩展第一层卷积 ──
    orig_weight = model.conv1.weight.data.clone()  # (64, 3, 7, 7)
    model.conv1 = nn.Conv2d(
        config.NUM_BANDS, 64,
        kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.conv1.weight.data = expand_first_layer(orig_weight)

    # ── 替换分类头 ──
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[models] ResNet-50 构建完成，参数量={total:.1f}M")
    return model


# ─── ViT-Small/16 ─────────────────────────────────────────────────────────────

def build_vit_small(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    """
    构建 ViT-Small/16，输入 224×224（与预训练完全一致）。

    策略
    ----
    - 输入尺寸 224×224 → pos_embed (1, 197, 384) 无需修改，完全复用预训练权重。
    - patch_embed.proj 通道从 3 扩展到 13：
        Step1: 额外加载标准 3 通道预训练模型，取出 proj.weight (384, 3, 16, 16)
        Step2: 创建 13 通道版本（timm 会用 kaiming_uniform 随机初始化 proj）
        Step3: 用 expand_first_layer 生成物理映射权重，覆盖 kaiming 初始化
    - 分类头替换为 num_classes 输出（timm 通过 num_classes 参数自动处理）
    """
    import timm

    # Step 1: 取出标准 3 通道预训练的 patch_embed.proj 权重
    ref_model = timm.create_model(
        "vit_small_patch16_224", pretrained=pretrained, in_chans=3, num_classes=num_classes
    )
    orig_proj_weight = ref_model.patch_embed.proj.weight.data.clone()  # (384, 3, 16, 16)
    del ref_model  # 释放内存

    # Step 2: 创建 13 通道模型（in_chans=13 使 timm 随机初始化 proj）
    model = timm.create_model(
        "vit_small_patch16_224", pretrained=pretrained, in_chans=13, num_classes=num_classes
    )

    # Step 3: 用物理映射权重替换 patch_embed.proj（覆盖 kaiming 随机初始化）
    new_proj_weight = expand_first_layer(orig_proj_weight)  # (384, 13, 16, 16)
    model.patch_embed.proj.weight.data = new_proj_weight

    # 验证位置编码未被修改
    assert model.pos_embed.shape == (1, 197, 384), \
        f"pos_embed 形状异常：{model.pos_embed.shape}，期望 (1, 197, 384)"

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[models] ViT-Small/16 构建完成，参数量={total:.1f}M")
    print(f"         pos_embed.shape={tuple(model.pos_embed.shape)}  OK")
    return model


# ─── 统一入口 ─────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    """
    统一模型构建入口。

    Parameters
    ----------
    model_name : str
        "resnet50" 或 "vit_small"
    """
    if model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_small":
        return build_vit_small(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"未知模型名称：{model_name!r}，可选：'resnet50', 'vit_small'")
