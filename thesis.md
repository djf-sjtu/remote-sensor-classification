# Remote Sensor Image Classification Using Deep Learning: CNN or Transformer?

**DTS311TC Final Year Project**

In Partial Fulfillment of the Requirements for the Degree of Bachelor of Engineering

School of AI and Advanced Computing

Xi'an Jiaotong-Liverpool University

April 2026

---

## Abstract

Remote sensing image classification plays a critical role in environmental monitoring, urban planning, agricultural management, and disaster response. With the rapid development of deep learning, convolutional neural networks (CNNs) have become the dominant approach for extracting spatial and spectral features from satellite imagery. More recently, Vision Transformers (ViTs), which leverage self-attention mechanisms to capture global contextual relationships, have emerged as a promising alternative to CNNs in computer vision tasks. However, a key question remains unanswered: which architecture is more suitable for multi-spectral remote sensing image classification?

This study provides a rigorous, controlled comparison between ResNet-50, a representative CNN architecture, and ViT-Small/16, a Transformer-based model, for land cover classification using the EuroSAT dataset. The EuroSAT dataset comprises 27,000 Sentinel-2 satellite images spanning 10 land use and land cover classes, each containing 13 spectral bands. To adapt ImageNet-pretrained models (originally designed for 3-channel RGB input) to 13-band multispectral data, a physics-aware band expansion strategy is proposed that maps Sentinel-2 bands to their spectrally corresponding RGB channels. Both models are fine-tuned under strictly identical training conditions, including the AdamW optimizer, cosine annealing learning rate schedule, and identical data augmentation, ensuring that any performance difference is attributable solely to the architectural design.

Experimental results across three independent runs demonstrate that both architectures achieve near-identical overall performance, with ResNet-50 attaining an overall accuracy of 98.98% and ViT-Small/16 achieving 98.93%. Notably, ViT-Small/16 exhibits greater stability across runs (lower standard deviation) and excels at classifying structurally homogeneous classes such as SeaLake and Industrial areas, while ResNet-50 shows marginal advantages on texture-rich vegetation classes. These findings suggest that, under controlled conditions with physics-aware transfer learning, both CNN and Transformer architectures are highly effective for multispectral remote sensing classification, with each exhibiting complementary per-class strengths.

**Keywords:** remote sensing, land cover classification, convolutional neural network, Vision Transformer, ResNet-50, ViT, EuroSAT, transfer learning, multispectral imagery, Sentinel-2

---

## Acknowledgements

I would like to express my sincere gratitude to my supervisor for the invaluable guidance, patience, and support provided throughout this project. Their expertise in deep learning and remote sensing has been instrumental in shaping the direction and quality of this research.

I am grateful to Xi'an Jiaotong-Liverpool University and the School of AI and Advanced Computing for providing an excellent academic environment and the computational resources necessary for conducting the experiments in this study.

I would also like to thank my peers and classmates for the stimulating discussions and constructive feedback during the course of this project. Their insights and encouragement have contributed significantly to my understanding of the subject matter.

Finally, I wish to thank my family for their unwavering support and encouragement throughout my studies. Their belief in me has been a constant source of motivation.

---

## Contents

*(Table of Contents — to be auto-generated)*

---

## 1. Introduction

### 1.1 Problem Description and Background Information

Satellite remote sensing has transformed our ability to observe and monitor Earth's surface at scale. Modern Earth observation satellites such as Sentinel-2 provide free, open-access multispectral imagery, capturing data across 13 spectral bands that span from visible light through near-infrared (NIR) to shortwave infrared (SWIR). This rich spectral information reveals properties invisible to the human eye: healthy vegetation reflects strongly in near-infrared, water bodies absorb NIR radiation, and urban surfaces exhibit distinctive spectral signatures.

Land cover classification—the task of assigning each image to a category such as "forest," "residential area," or "farmland"—is essential for numerous real-world applications. Conservation organisations rely on accurate land cover maps to track deforestation. Urban planners use classified satellite imagery to monitor urban sprawl. Agricultural agencies assess crop health through spectral analysis, and emergency response teams leverage classified images during disaster management.

Traditional machine learning approaches for remote sensing classification relied heavily on handcrafted spectral features combined with classifiers like Support Vector Machines (SVMs) and Random Forests. While effective for small-scale applications, these methods required significant domain expertise for feature engineering and struggled to generalise across different sensors and geographical regions. Deep learning has fundamentally changed this landscape. Convolutional Neural Networks (CNNs) can automatically learn hierarchical features from raw data, and architectures such as ResNet have become standard backbones for remote sensing tasks. More recently, Vision Transformers (ViTs), originally developed for natural language processing, have shown competitive performance in computer vision by modelling global contextual relationships.

However, a significant gap exists in the current literature: most existing studies compare CNNs and Transformers using different datasets, preprocessing pipelines, or training configurations, making it difficult to attribute performance differences solely to architectural design. Furthermore, many studies operate exclusively on RGB imagery, discarding the rich spectral information available in multispectral sensors. This project addresses both gaps by providing a controlled, fair comparison of ResNet-50 and ViT-Small/16 on the full 13-band EuroSAT dataset, using a physics-aware band expansion strategy and identical training protocols.

### 1.2 Aims and Objectives

The primary aim of this project is to conduct a fair and reproducible comparison between a CNN-based architecture (ResNet-50) and a Transformer-based architecture (ViT-Small/16) for land cover classification on multispectral satellite imagery. By maintaining strictly identical experimental conditions—with the sole variable being the model architecture—this study seeks to isolate and understand the inherent strengths and weaknesses of each approach for remote sensing tasks.

The specific objectives of this project are as follows:

1. **Data pipeline implementation**: Develop a complete data processing pipeline capable of reading 13-band Sentinel-2 GeoTIFF images, computing per-channel normalisation statistics, and applying domain-appropriate spatial augmentations for remote sensing data.

2. **Physics-aware band expansion**: Design and implement a physics-aware band expansion strategy that maps ImageNet-pretrained 3-channel weights to 13-channel input, leveraging the spectral correspondence between Sentinel-2 bands and RGB wavelengths.

3. **Controlled model fine-tuning**: Fine-tune both ResNet-50 and ViT-Small/16 under identical training conditions, including the same optimizer, learning rate schedule, number of epochs, batch size, loss function, and data augmentation strategy.

4. **Comprehensive evaluation**: Evaluate both models using multiple complementary metrics: Overall Accuracy (OA), Macro-F1 score, per-class F1 scores, confusion matrices, and inference speed.

5. **Statistical validation**: Repeat each experiment with three different random seeds and report results as mean ± standard deviation to ensure statistical reliability.

6. **Per-class analysis**: Conduct detailed per-class performance analysis to identify which land cover types benefit from CNN's local feature extraction versus Transformer's global attention mechanism.

The scope of this study is intentionally focused to ensure rigour. The comparison is limited to two specific architectures (one CNN, one Transformer) on a single dataset (EuroSAT). Only supervised transfer learning from ImageNet pre-trained weights is considered; training from scratch, semi-supervised learning, and self-supervised pre-training are beyond the scope of this work. Hybrid architectures such as Swin Transformer or ConvNeXt are not included in the comparison but are discussed as directions for future work.

---

## 2. Literature Review

Remote sensing image classification has undergone a remarkable evolution over the past two decades, driven largely by advances in machine learning and, more recently, deep learning. This chapter reviews the key developments that have shaped the field, from traditional feature engineering approaches to the emergence of Vision Transformers, and identifies the research gap that motivates the present study.

### 2.1 Traditional Machine Learning in Remote Sensing

Before the advent of deep learning, remote sensing classification relied predominantly on pixel-level or object-based methods using handcrafted features. Spectral indices such as the Normalised Difference Vegetation Index (NDVI) and the Normalised Difference Water Index (NDWI), along with texture features derived from the Grey-Level Co-occurrence Matrix (GLCM), were commonly fed into classifiers such as Support Vector Machines (SVMs) [1], Random Forests, and k-Nearest Neighbours (k-NN). While these methods achieved reasonable accuracy for well-defined classification tasks, they suffered from several limitations: they required substantial domain expertise for feature engineering, generalised poorly across different sensors and geographical regions, and struggled to capture the complex, nonlinear relationships between spatial and spectral information inherent in multispectral imagery.

### 2.2 Convolutional Neural Networks for Remote Sensing

The deep learning revolution in computer vision, sparked by AlexNet's breakthrough performance on the ImageNet [2] Large Scale Visual Recognition Challenge (ILSVRC) [3], rapidly extended to remote sensing applications. Castelluccio et al. [4] demonstrated that fine-tuned CNNs (GoogLeNet, CaffeNet) significantly outperformed handcrafted feature-based approaches for land use classification on the UC Merced dataset. Zhu et al. [5] provided a comprehensive review documenting the rapid adoption of CNNs across remote sensing tasks including scene classification, object detection, and semantic segmentation. The key advantage of CNNs lies in their ability to automatically learn hierarchical spatial features through convolutional filters: early layers detect low-level features such as edges and textures, while deeper layers capture increasingly complex and abstract patterns, eliminating the need for manual feature engineering.

### 2.3 ResNet and Deep Residual Learning

A critical milestone in CNN development was the introduction of deep residual learning by He et al. [6]. By adding skip (shortcut) connections that allow gradients to flow directly through identity mappings, ResNet solved the vanishing gradient problem and enabled the training of networks exceeding 50 layers in depth. ResNet-50, with its balance of depth, representational capacity, and computational efficiency, quickly became the de facto standard backbone for a wide range of computer vision tasks, including remote sensing image classification. The architecture's built-in inductive biases—local receptive fields, translation equivariance, and hierarchical feature extraction—are inherently well-suited to spatial pattern recognition in satellite imagery, where local textures and spatial structures carry discriminative information.

### 2.4 The Transformer Revolution in Computer Vision

The Transformer architecture, introduced by Vaswani et al. [7] for natural language processing, relies entirely on self-attention mechanisms rather than convolution. Each element in the input sequence attends to every other element, enabling the model to capture long-range dependencies without the locality constraints of convolutional filters. Dosovitskiy et al. [8] adapted this paradigm to computer vision with the Vision Transformer (ViT), which divides an input image into fixed-size patches (e.g., 16×16 pixels), linearly embeds each patch, and processes the resulting sequence with a standard Transformer encoder. Crucially, ViT contains minimal inductive bias for images—it does not assume locality or translation equivariance—and instead learns spatial relationships entirely from data through positional embeddings and attention patterns. While this architectural flexibility is powerful, it makes ViT inherently data-hungry; Dosovitskiy et al. showed that ViT requires large-scale pre-training (e.g., on ImageNet-21k or JFT-300M) to match CNN performance. Touvron et al. [9] subsequently addressed this data efficiency challenge with DeiT (Data-efficient Image Transformer), demonstrating that careful training strategies, including knowledge distillation, can enable competitive ViT performance with standard ImageNet-1k pre-training alone.

### 2.5 Transformers in Remote Sensing

The application of Transformers to remote sensing has grown rapidly since 2021. Bazi et al. [10] were among the first to apply ViT to remote sensing scene classification, achieving competitive results across multiple benchmark datasets. He et al. [11] proposed a spatial-spectral Transformer specifically designed for hyperspectral image classification, demonstrating that self-attention mechanisms can effectively model both spatial and spectral dependencies. Aleissaee et al. [12] provided a comprehensive survey documenting the accelerating adoption of Transformers across eight remote sensing domains, including land use/land cover (LULC) classification, segmentation, change detection, and object detection. Their quantitative analysis showed that Transformers achieve higher accuracy in LULC classification tasks, with more stable performance compared to traditional approaches. The global receptive field of Transformers is particularly appealing for remote sensing, where contextual information from distant spatial regions—such as a river's relationship to surrounding vegetation—can carry significant discriminative value.

### 2.6 The EuroSAT Dataset and Transfer Learning for Remote Sensing

Helber et al. [13] introduced EuroSAT, a benchmark dataset comprising 27,000 geo-referenced Sentinel-2 [14] satellite images spanning 10 land use and land cover classes across 34 European countries. The dataset provides both RGB (3-band) and full multispectral (13-band) versions, with reported classification accuracies exceeding 98% for RGB-only deep learning models. EuroSAT has since become a widely adopted benchmark for evaluating deep learning approaches on satellite imagery. Neumann et al. [15] explored in-domain representation learning strategies for remote sensing, investigating how transfer learning from ImageNet can be effectively adapted to the unique characteristics of satellite data—including the challenge of adapting 3-channel pre-trained models to multispectral inputs with more than three bands.

### 2.7 Research Gap

Despite the growing body of work on both CNNs and Transformers for remote sensing, systematic and controlled comparisons between these two paradigms remain scarce. Maurício et al. [16] conducted a comprehensive literature review of CNN versus ViT comparisons for image classification, but noted that the majority of such studies focus on natural images (e.g., ImageNet, CIFAR) rather than domain-specific applications. In remote sensing specifically, most existing studies either use different datasets, preprocessing pipelines, or training configurations when evaluating CNNs versus Transformers, making it impossible to isolate the effect of architecture alone. Furthermore, the challenge of adapting 3-channel ImageNet-pretrained models to 13-band multispectral input is often handled by either discarding non-RGB bands or through naive channel replication, thereby losing valuable spectral information. This project addresses both gaps simultaneously: it provides a fair, controlled comparison under identical experimental conditions, and introduces a physics-aware band expansion strategy that preserves pretrained knowledge while accommodating all 13 Sentinel-2 spectral bands.

---

## 3. Methodology and Experiment

### 3.1 Methodology

#### 3.1.1 Overall System Architecture

The overall system architecture of this project is designed around a central principle: both models must share every component of the experimental pipeline except the model architecture itself. This ensures that any observed performance differences can be attributed solely to the architectural design rather than to differences in data handling, training strategy, or evaluation protocol.

The complete pipeline proceeds as follows. First, raw Sentinel-2 GeoTIFF images (13 bands, 64×64 pixels) are read and converted to floating-point reflectance values. The dataset is then split into training, validation, and test sets using stratified sampling to preserve class distribution. Per-channel normalisation statistics (mean and standard deviation) are computed on the training set only, preventing data leakage. During training, images are resized to 224×224 pixels, spatially augmented (training set only), and normalised. The processed images are fed into either ResNet-50 or ViT-Small/16, both of which have been adapted from 3-channel ImageNet pre-trained weights to 13-channel input via a physics-aware band expansion strategy. Both models are trained using identical optimizers, learning rate schedules, loss functions, and hyperparameters. After training, the best checkpoint (selected by validation Macro-F1) is loaded and evaluated on the held-out test set. Each experiment is repeated with three random seeds, and results are aggregated as mean ± standard deviation.

Figure 1 illustrates the overall system architecture.

> **Figure 1.** Overall system architecture. Raw GeoTIFF images pass through the data pipeline (read → normalise → resize → augment) before being fed into either ResNet-50 or ViT-Small/16 with physics-aware band expansion. Training uses identical conditions for both models. Evaluation produces OA, Macro-F1, per-class F1, confusion matrices, and inference speed metrics, aggregated across three seeds.

#### 3.1.2 Dataset Description

This study uses the EuroSAT dataset [13], a widely adopted benchmark for land use and land cover (LULC) classification from satellite imagery. The dataset comprises 27,000 geo-referenced image patches acquired by the Sentinel-2 satellite [14], which is part of the European Space Agency's (ESA) Copernicus Earth observation programme. Each image patch covers an area of approximately 640 m × 640 m at 10 m spatial resolution and is stored as a GeoTIFF file containing all 13 Sentinel-2 spectral bands at a resolution of 64×64 pixels.

The 13 spectral bands of Sentinel-2 span a wide range of the electromagnetic spectrum, from coastal aerosol (443 nm) to shortwave infrared (2190 nm). Table 1 summarises the band specifications.

**Table 1.** Sentinel-2 spectral band specifications used in this study.

| Band Index | Sentinel-2 Band | Central Wavelength (nm) | Spatial Resolution (m) | Description |
|:---:|:---:|:---:|:---:|:---|
| 0 | B1 | 443 | 60 | Coastal aerosol |
| 1 | B2 | 490 | 10 | Blue |
| 2 | B3 | 560 | 10 | Green |
| 3 | B4 | 665 | 10 | Red |
| 4 | B5 | 705 | 20 | Vegetation red edge |
| 5 | B6 | 740 | 20 | Vegetation red edge |
| 6 | B7 | 783 | 20 | Vegetation red edge |
| 7 | B8 | 842 | 10 | Near-infrared (NIR) |
| 8 | B8A | 865 | 20 | Narrow NIR |
| 9 | B9 | 945 | 60 | Water vapour |
| 10 | B10 | 1375 | 60 | Cirrus |
| 11 | B11 | 1610 | 20 | SWIR-1 |
| 12 | B12 | 2190 | 20 | SWIR-2 |

The dataset is organised into 10 land cover classes. Table 2 presents the class distribution.

**Table 2.** EuroSAT class distribution.

| Class | Number of Samples |
|:---|:---:|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |
| **Total** | **27,000** |

The dataset exhibits mild class imbalance, with Pasture containing 2,000 samples while several other classes contain 3,000. This imbalance motivates the use of Macro-F1—which weights all classes equally regardless of sample count—as the primary evaluation metric, rather than relying solely on Overall Accuracy.

#### 3.1.3 Data Pipeline

The data pipeline encompasses four stages: data reading, dataset splitting, normalisation statistics computation, and runtime preprocessing with augmentation.

**Data Reading.** Each GeoTIFF image is read using the rasterio library, yielding a uint16 array of shape (13, 64, 64). The raw digital numbers are divided by 10,000 to convert them to approximate surface reflectance values in the range [0, ~1]. This scaling is physically meaningful, as Sentinel-2 Level-2A products encode reflectance scaled by a factor of 10,000.

**Dataset Splitting.** A stratified 70/10/20 train/validation/test split is performed using scikit-learn's `StratifiedShuffleSplit` with a fixed seed of 42. The splitting proceeds in two stages: first, 20% of the data is separated as the test set; then, from the remaining 80%, 12.5% (equivalent to 10% of the total) is allocated to the validation set, with the rest forming the training set. Stratification ensures that each class maintains its original proportion across all three splits. Table 3 summarises the split sizes.

**Table 3.** Dataset split sizes.

| Split | Number of Samples | Percentage |
|:---|:---:|:---:|
| Training | 18,900 | 70% |
| Validation | 2,700 | 10% |
| Test | 5,400 | 20% |

**Normalisation Statistics.** The per-channel mean and standard deviation for all 13 bands are computed exclusively on the training set to prevent data leakage. This computation uses the Welford online algorithm, a single-pass, numerically stable method that processes one image at a time without loading the entire dataset into memory. Statistics are computed on the original 64×64 pixel resolution (before resizing) to avoid interpolation-induced errors, and are cached to disk for reproducibility.

**Runtime Preprocessing and Augmentation.** At runtime, each image is first converted to a PyTorch tensor and resized from 64×64 to 224×224 pixels using bilinear interpolation with antialiasing. This upsampling is necessary because both ResNet-50 and ViT-Small/16 are designed for 224×224 input. For the training set, spatial augmentations are applied, including random horizontal flip (p=0.5), random vertical flip (p=0.5), random 90-degree rotation (p=0.5), and random resized crop with a scale range of [0.75, 1.0]. These augmentations are specifically chosen for remote sensing: satellite images have no canonical orientation (unlike natural photographs), so all four 90-degree rotations are equally valid. No colour or spectral augmentation is applied, to preserve the physical meaning of the spectral bands. After augmentation, all images are normalised using the pre-computed per-channel mean and standard deviation: x = (x − μ) / σ. The validation and test sets undergo only resizing and normalisation, without any augmentation.

#### 3.1.4 Physics-Aware Band Expansion Strategy

**The Challenge.** Both ResNet-50 and ViT-Small/16 are pre-trained on ImageNet, which consists of 3-channel RGB images. The EuroSAT dataset, however, contains 13 spectral bands. A naive approach would be to either discard 10 bands and use only the RGB subset, or to randomly initialise the first layer for 13 channels and discard all pre-trained knowledge in that layer. Neither approach is satisfactory: the former loses valuable spectral information (NIR, SWIR, and red-edge bands are critical for vegetation and water classification), while the latter undermines the benefits of transfer learning.

**The Solution.** This study proposes a physics-aware initialisation strategy that maps each Sentinel-2 band to the most spectrally similar ImageNet RGB channel. For bands with a direct correspondence in the visible spectrum (B2→Blue, B3→Green, B4→Red), the pre-trained weights are directly copied from the matching RGB channel. For red-edge bands (B5, B6) that are spectrally closest to the red channel, the red channel weights are copied. For all remaining bands that fall outside the visible spectrum (NIR, SWIR, water vapour, cirrus, and coastal aerosol), the RGB three-channel mean is used as the initialisation. Table 4 details the complete mapping strategy.

**Table 4.** Physics-aware band-to-RGB mapping strategy.

| Band Index | Sentinel-2 Band | Wavelength | Strategy | Source | Rationale |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | B1 (Aerosol) | 443 nm | mean | RGB mean | No direct RGB correspondence |
| 1 | B2 (Blue) | 490 nm | copy | Blue channel | Direct spectral match |
| 2 | B3 (Green) | 560 nm | copy | Green channel | Direct spectral match |
| 3 | B4 (Red) | 665 nm | copy | Red channel | Direct spectral match |
| 4 | B5 (Red Edge) | 705 nm | copy | Red channel | Nearest neighbour in spectrum |
| 5 | B6 (Red Edge) | 740 nm | copy | Red channel | Nearest neighbour in spectrum |
| 6 | B7 (Red Edge) | 783 nm | mean | RGB mean | NIR boundary, no direct match |
| 7 | B8 (NIR) | 842 nm | mean | RGB mean | Beyond visible spectrum |
| 8 | B8A (Narrow NIR) | 865 nm | mean | RGB mean | Beyond visible spectrum |
| 9 | B9 (Water Vapour) | 945 nm | mean | RGB mean | Beyond visible spectrum |
| 10 | B10 (Cirrus) | 1375 nm | mean | RGB mean | Beyond visible spectrum |
| 11 | B11 (SWIR-1) | 1610 nm | mean | RGB mean | Beyond visible spectrum |
| 12 | B12 (SWIR-2) | 2190 nm | mean | RGB mean | Beyond visible spectrum |

**Scale Factor.** After expansion from 3 to 13 channels, the convolution output magnitude would increase by approximately 13/3 ≈ 4.33× (since more input channels contribute to the weighted sum). To preserve the activation magnitude and maintain compatibility with subsequent pre-trained layers, all expanded weights are multiplied by a scale factor of 3/13 ≈ 0.231. This ensures that the output distribution of the first layer remains similar to what the downstream pre-trained layers expect, facilitating effective transfer learning.

The `expand_first_layer` function implements this strategy in a model-agnostic manner and is applied identically to both ResNet-50 (on the `conv1` layer) and ViT-Small/16 (on the `patch_embed.proj` layer), ensuring a fair comparison.

The algorithm proceeds as follows:

```
function expand_first_layer(original_weight, band_init_map):
    Input: original_weight of shape (out_channels, 3, kH, kW)
    Output: new_weight of shape (out_channels, 13, kH, kW)

    rgb_mean = mean(original_weight, dim=1)  // shape: (out_channels, kH, kW)
    new_weight = zeros(out_channels, 13, kH, kW)

    for each (channel_index, (strategy, source_index)) in band_init_map:
        if strategy == 'copy':
            new_weight[:, channel_index] = original_weight[:, source_index]
        else:  // strategy == 'mean'
            new_weight[:, channel_index] = rgb_mean

    scale = 3.0 / 13.0
    new_weight = new_weight * scale

    return new_weight
```

#### 3.1.5 ResNet-50 Architecture

ResNet-50 [6] is a 50-layer deep convolutional neural network organised into five stages, introduced by He et al. to address the vanishing gradient problem through residual (skip) connections. The core building block of ResNet-50 is the bottleneck block, which consists of three sequential convolutions—a 1×1 convolution for channel reduction, a 3×3 convolution for spatial processing, and a 1×1 convolution for channel expansion—each followed by batch normalisation and ReLU activation. A shortcut connection adds the input directly to the output, enabling gradient flow through identity mappings and allowing the network to learn residual functions rather than complete transformations.

Table 5 summarises the ResNet-50 architecture as configured for this study.

**Table 5.** ResNet-50 architecture summary for 13-band input.

| Stage | Layers | Output Size | Channels | Key Operation |
|:---|:---:|:---:|:---:|:---|
| Input | — | 224×224 | 13 | — |
| Conv1 + BN + ReLU + MaxPool | 1 conv | 56×56 | 64 | 7×7 conv, stride 2; 3×3 max pool |
| Conv2_x | 3 bottleneck blocks | 56×56 | 256 | 1×1, 3×3, 1×1 convolutions |
| Conv3_x | 4 bottleneck blocks | 28×28 | 512 | Stride-2 downsampling |
| Conv4_x | 6 bottleneck blocks | 14×14 | 1024 | Stride-2 downsampling |
| Conv5_x | 3 bottleneck blocks | 7×7 | 2048 | Stride-2 downsampling |
| Global Average Pooling | 1 | 1×1 | 2048 | Adaptive average pooling |
| Fully Connected | 1 linear | — | 10 | Classification head |

**Modifications for this study.** Two layers are modified from the standard ImageNet-pretrained ResNet-50. First, the initial convolutional layer `conv1` is replaced from `Conv2d(3, 64, 7, 7, stride=2, padding=3)` to `Conv2d(13, 64, 7, 7, stride=2, padding=3)`, and the expanded weights are initialised using the physics-aware band expansion strategy described in Section 3.1.4. Second, the final classification head is replaced from `Linear(2048, 1000)` to `Linear(2048, 10)`, with weights initialised using a normal distribution (mean=0, std=0.01) and biases set to zero. All other layers (Conv2_x through Conv5_x, including batch normalisation parameters) retain their ImageNet pre-trained weights. The total parameter count is approximately 23.56 million.

**Inductive biases.** ResNet's convolutional architecture embeds strong inductive biases: locality (each filter operates on a small spatial neighbourhood), translation equivariance (the same filter is applied at all spatial positions), and hierarchical feature extraction (spatial resolution decreases while channel depth increases through the network). These biases are well-suited to spatial pattern recognition but limit the model's ability to capture long-range dependencies within a single layer.

#### 3.1.6 ViT-Small/16 Architecture

The Vision Transformer (ViT) [8] adapts the Transformer architecture from natural language processing to computer vision. Rather than processing pixels through convolution, ViT divides the input image into fixed-size non-overlapping patches, projects each patch into a vector embedding, and processes the resulting sequence with a standard Transformer encoder.

**Patch Embedding.** The input image of shape (13, 224, 224) is divided into non-overlapping 16×16 patches, yielding a grid of 14×14 = 196 patches. Each patch of shape (13, 16, 16) is flattened and linearly projected to a 384-dimensional embedding vector via a convolutional projection layer `patch_embed.proj: Conv2d(13, 384, 16, 16, stride=16)`. A learnable class token [CLS] is prepended to the sequence, and learnable positional embeddings of shape (1, 197, 384) are added element-wise to encode spatial information.

**Transformer Encoder.** The resulting sequence of 197 tokens (196 patches + 1 class token) is processed by 12 identical Transformer encoder blocks. Each block consists of two sub-layers: multi-head self-attention (MHSA) and a feed-forward MLP, each preceded by layer normalisation and followed by a residual connection. The MHSA uses 6 attention heads, each with dimension 384/6 = 64. The attention computation for each head follows: Attention(Q, K, V) = softmax(QK^T / √d_k)V, where Q, K, and V are the query, key, and value matrices. This mechanism gives ViT a global receptive field from the very first layer: every patch attends to every other patch, enabling the model to capture long-range spatial dependencies directly.

Table 6 summarises the ViT-Small/16 architecture.

**Table 6.** ViT-Small/16 architecture specifications.

| Component | Specification |
|:---|:---|
| Patch size | 16 × 16 |
| Number of patches | 14 × 14 = 196 |
| Embedding dimension | 384 |
| Number of Transformer blocks | 12 |
| Number of attention heads | 6 |
| Head dimension | 384 / 6 = 64 |
| MLP hidden dimension | 384 × 4 = 1536 |
| Sequence length | 197 (196 patches + 1 CLS token) |
| Positional embedding | Learnable, shape (1, 197, 384) |
| Classification head | Linear(384, 10) |
| Total parameters | ~22.65 million |

**Modifications for this study.** The ViT-Small/16 model is constructed using the timm library [18]. The patch embedding projection layer is expanded from `Conv2d(3, 384, 16, 16)` to `Conv2d(13, 384, 16, 16)` using the same physics-aware band expansion strategy applied to ResNet-50. Critically, because the input spatial resolution remains 224×224 (matching the pre-training resolution), the positional embeddings (1, 197, 384) are unchanged and fully reused from the pre-trained model. The classification head is automatically configured to output 10 classes.

**Inductive biases.** Unlike CNNs, ViT has minimal built-in inductive bias for images. It does not assume locality or translation equivariance; spatial relationships are learned entirely from data through positional embeddings and attention patterns. This makes ViT more flexible but potentially data-hungry [9], which is why ImageNet pre-training is critical for achieving competitive performance on the relatively small EuroSAT dataset.

#### 3.1.7 Model Comparison Summary

Table 7 provides a side-by-side comparison of the two architectures as configured in this study.

**Table 7.** Architectural comparison of ResNet-50 and ViT-Small/16.

| Property | ResNet-50 | ViT-Small/16 |
|:---|:---|:---|
| Architecture type | CNN | Transformer |
| Total parameters | ~23.56 M | ~22.65 M |
| First layer (modified) | Conv2d(13, 64, 7×7) | Conv2d(13, 384, 16×16) |
| Feature extraction | Hierarchical convolutions | Global self-attention |
| Receptive field | Local, grows with depth | Global from layer 1 |
| Inductive bias | Locality, translation equivariance | Minimal (learned from data) |
| Spatial downsampling | Progressive (224→56→28→14→7) | Single step (224→14×14 patches) |
| Pre-trained source | torchvision (ImageNet1K_V2) | timm [18] (ImageNet1K) |
| Classification head | Linear(2048, 10) | Linear(384, 10) |
| Band expansion | conv1 weights expanded | patch_embed.proj weights expanded |

Based on the architectural differences, one might hypothesise that ResNet-50, with its strong local inductive biases and hierarchical feature extraction, should excel at classes characterised by fine-grained local textures (e.g., vegetation types), while ViT-Small/16, with its global attention mechanism, should perform better on classes where contextual information from distant regions is informative (e.g., distinguishing rivers from lakes, or identifying industrial areas based on surrounding land use patterns).

#### 3.1.8 Evaluation Metrics

To comprehensively assess and compare the two architectures, five complementary evaluation metrics are employed.

**Overall Accuracy (OA).** The proportion of correctly classified samples out of the total number of test samples. While intuitive, OA can be misleading for imbalanced datasets, as it is dominated by the majority classes.

**Macro-F1 Score.** The unweighted mean of per-class F1 scores, where F1 = 2 × Precision × Recall / (Precision + Recall). Macro-F1 treats all classes equally regardless of their sample count, making it a more informative metric than OA for datasets with class imbalance. In this study, Macro-F1 is used as the primary metric for model selection: the best checkpoint during training is saved based on the highest validation Macro-F1.

**Per-Class F1 Score.** Individual F1 scores computed for each of the 10 land cover classes. This metric enables fine-grained analysis of which land cover types each architecture handles well or poorly, revealing the complementary strengths of CNN and Transformer approaches.

**Confusion Matrix.** A 10×10 matrix where entry (i, j) represents the proportion of samples from true class i predicted as class j. Row-normalised confusion matrices visualise the recall for each class along the diagonal and reveal systematic confusion patterns in the off-diagonal entries, providing insight into which classes are commonly misclassified and why.

**Inference Speed.** Measured as milliseconds per image on the test set, including data transfer to GPU and forward pass computation. This metric provides a practical measure of computational efficiency that is relevant for deployment in operational remote sensing workflows.

**Statistical Validity.** Each experiment is repeated with three random seeds (0, 1, 2), with results reported as mean ± standard deviation. The data split seed (42) is fixed across all experiments, ensuring that all models are trained and evaluated on exactly the same data subsets, and that the only source of variability is the random initialisation and training order.

### 3.2 Experiment

#### 3.2.1 Hardware and Software Environment

All experiments were conducted on a single workstation. Table 8 summarises the hardware and software configuration.

**Table 8.** Experimental environment.

| Component | Specification |
|:---|:---|
| GPU | NVIDIA GPU with CUDA support |
| Operating System | Windows |
| Python | 3.x |
| PyTorch | Latest stable release |
| torchvision | Compatible version with PyTorch |
| timm | Latest stable release |
| rasterio | Latest stable release |
| scikit-learn | Latest stable release |
| CUDA | Compatible version with GPU |

Mixed-precision training (AMP) was enabled to accelerate computation and reduce GPU memory usage. All models were trained and evaluated on the same hardware to ensure fair timing comparisons.

#### 3.2.2 Training Configuration

Both models are trained under strictly identical hyperparameter settings to ensure a fair comparison. Table 9 details the complete training configuration.

**Table 9.** Shared training hyperparameters.

| Hyperparameter | Value |
|:---|:---|
| Optimizer | AdamW [17] |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 0.05 |
| Betas | (0.9, 0.999) |
| LR scheduler | Cosine Annealing |
| Minimum LR (η_min) | 1 × 10⁻⁶ |
| Epochs | 10 |
| Batch size (training) | 64 |
| Batch size (validation/test) | 128 |
| Loss function | CrossEntropyLoss |
| Mixed precision | Enabled (torch.amp) |
| Gradient clipping | max_norm = 1.0 |
| Best checkpoint metric | Validation Macro-F1 |
| Number of data workers | 4 |

The AdamW optimizer [17] is chosen for its decoupled weight decay regularisation, which has been shown to improve generalisation compared to standard Adam with L2 regularisation. The cosine annealing schedule smoothly decreases the learning rate from 1×10⁻⁴ to 1×10⁻⁶ over the training period, facilitating convergence in the later stages of training. Gradient clipping with a maximum norm of 1.0 is applied to prevent gradient explosion, which can occur particularly with Transformer architectures.

**Checkpoint Selection.** At the end of each epoch, the model is evaluated on the validation set. If the current validation Macro-F1 exceeds the previous best, the model state is saved as the best checkpoint. After training completes, this best checkpoint is loaded for final evaluation on the held-out test set.

**Reproducibility.** At the start of each experiment, the random states of Python, NumPy, PyTorch, and CUDA are fixed using the experiment seed. Additionally, `torch.backends.cudnn.deterministic` is set to `True` and `torch.backends.cudnn.benchmark` is set to `False` to ensure deterministic behaviour across runs.

#### 3.2.3 Experiment Protocol

The experimental protocol follows a systematic design with 2 models × 3 seeds = 6 total experiments. Table 10 presents the experiment matrix.

**Table 10.** Experiment matrix.

| | Seed 0 | Seed 1 | Seed 2 |
|:---|:---:|:---:|:---:|
| ResNet-50 | Run 1 | Run 2 | Run 3 |
| ViT-Small/16 | Run 4 | Run 5 | Run 6 |

All six experiments share the same dataset split (split seed = 42) and the same pre-computed normalisation statistics, ensuring that every model is trained and tested on identical data. The experiment seed controls the random weight initialisation of the classification head, the data loading order (shuffle), and the stochastic augmentation applied during training.

Each individual experiment proceeds through the following stages:

1. **Seed initialisation**: Fix random states for Python, NumPy, PyTorch, and CUDA.
2. **DataLoader construction**: Build training, validation, and test data loaders from the shared split.
3. **Model construction**: Build the model (ResNet-50 or ViT-Small/16) with physics-aware band expansion from ImageNet pre-trained weights.
4. **Training**: Train for 10 epochs, logging per-epoch metrics (train loss, train accuracy, validation loss, validation accuracy, validation Macro-F1, and learning rate) to a CSV file.
5. **Checkpoint loading**: Load the best checkpoint (highest validation Macro-F1).
6. **Test evaluation**: Evaluate on the held-out test set, computing OA, Macro-F1, per-class F1, confusion matrix, inference speed, and parameter count.
7. **Visualisation**: Generate training curve plots and confusion matrix heatmaps.
8. **Result serialisation**: Save all metrics to a JSON file for aggregation and checkpoint-based recovery.

The system supports checkpoint-based recovery: if a run has already been completed (indicated by the existence of its result JSON file), it is automatically skipped, allowing the experiment to resume from interruptions without redundant computation.

#### 3.2.4 Training Monitoring

Figures 2 and 3 present the training curves for ResNet-50 (seed 0) and ViT-Small/16 (seed 0), respectively.

![ResNet-50 Training Curves (Seed 0)](outputs/figures/resnet50_seed0_curves.png)

> **Figure 2.** Training curves for ResNet-50 (seed 0). The four panels show: (top-left) train and validation loss, (top-right) train and validation accuracy, (bottom-left) validation Macro-F1, and (bottom-right) learning rate schedule. The dashed vertical line marks the epoch with the best validation Macro-F1.

![ViT-Small/16 Training Curves (Seed 0)](outputs/figures/vit_small_seed0_curves.png)

> **Figure 3.** Training curves for ViT-Small/16 (seed 0). Same layout as Figure 2.

Several observations can be drawn from the training dynamics:

**Convergence speed.** ResNet-50 achieves a training accuracy of 90.1% after the first epoch, while ViT-Small/16 reaches 92.6%. This indicates that ViT-Small/16 benefits from slightly faster initial convergence, possibly due to the larger effective receptive field provided by global self-attention from the first layer. However, both models converge rapidly, reaching over 98% validation accuracy within the first few epochs.

**Training vs. validation gap.** By epoch 10, ViT-Small/16 achieves a higher training accuracy (99.94%) compared to ResNet-50 (99.58%), suggesting that ViT-Small/16 fits the training data more tightly. However, both models achieve similar validation performance (ResNet-50: 99.15% val accuracy, val Macro-F1 = 0.9912; ViT-Small/16: 98.96% val accuracy, val Macro-F1 = 0.9891), indicating that ViT-Small/16's tighter fit to training data does not translate into superior generalisation on the validation set at this scale.

**Learning rate schedule.** The cosine annealing schedule smoothly reduces the learning rate from 1×10⁻⁴ to approximately 3.4×10⁻⁶ by epoch 10. Both models show the largest performance gains in the early epochs when the learning rate is highest, with diminishing improvements as the learning rate decreases. The best validation checkpoints for both models occur in the final epochs (epoch 9–10), where the low learning rate enables fine-grained parameter adjustments.

**Loss behaviour.** Both models exhibit a characteristic pattern where training loss continues to decrease throughout training, while validation loss plateaus or slightly increases in the later epochs. This mild divergence is typical of deep learning models and does not indicate significant overfitting, as the validation accuracy remains high and stable.

### 3.3 Results

#### 3.3.1 Overall Performance Comparison

Table 11 presents the aggregated test-set results for both models across three independent runs.

**Table 11.** Overall performance comparison (mean ± std over 3 runs).

| Metric | ResNet-50 | ViT-Small/16 |
|:---|:---:|:---:|
| Overall Accuracy | 0.9898 ± 0.0013 | 0.9893 ± 0.0008 |
| Macro-F1 | 0.9895 ± 0.0014 | 0.9890 ± 0.0008 |
| Parameters (M) | 23.56 | 22.65 |
| Inference Speed (ms/img) | 15.54 | 15.72 |
| Training Time (min/run) | 60.4 | 73.5 |

![Overall Performance Comparison](outputs/figures/comparison_metrics.png)

> **Figure 4.** Overall Accuracy and Macro-F1 comparison between ResNet-50 and ViT-Small/16. Error bars represent one standard deviation across three runs. Both models achieve near-identical performance, with ResNet-50 showing a marginal advantage of 0.05 percentage points in OA.

The most striking finding is the near-identical performance of both architectures. ResNet-50 achieves a marginally higher overall accuracy (98.98% vs. 98.93%) and Macro-F1 (98.95% vs. 98.90%), but the differences are well within one standard deviation and are not statistically significant. This suggests that, under controlled conditions with physics-aware transfer learning from ImageNet, both CNN and Transformer architectures are equally effective for multispectral remote sensing classification on the EuroSAT dataset.

However, an important distinction emerges in the stability of results. ViT-Small/16 exhibits notably lower variance across runs, with standard deviations of 0.0008 for both OA and Macro-F1, compared to 0.0013 and 0.0014 for ResNet-50. This indicates that ViT-Small/16 is more robust to random initialisation and training order variations, which may be attributable to the global self-attention mechanism providing more stable feature representations than locally-constrained convolutions.

In terms of computational resources, both models have comparable parameter counts (23.56M vs. 22.65M) and nearly identical inference speeds (15.54 vs. 15.72 ms/img). The average training time per run is approximately 60 minutes for ResNet-50 and 74 minutes for ViT-Small/16, reflecting the higher computational cost of self-attention operations during training.

#### 3.3.2 Per-Class Performance Analysis

Table 12 presents the per-class F1 scores for all six experimental runs.

**Table 12.** Per-class F1 scores for all experimental runs.

| Class | ResNet Seed 0 | ResNet Seed 1 | ResNet Seed 2 | ViT Seed 0 | ViT Seed 1 | ViT Seed 2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| AnnualCrop | 0.9849 | 0.9824 | 0.9773 | 0.9782 | 0.9712 | 0.9774 |
| Forest | 0.9967 | 0.9983 | 0.9950 | 0.9967 | 0.9983 | 0.9983 |
| HerbaceousVeg. | 0.9900 | 0.9875 | 0.9875 | 0.9834 | 0.9866 | 0.9875 |
| Highway | 0.9900 | 0.9861 | 0.9880 | 0.9881 | 0.9890 | 0.9890 |
| Industrial | 0.9910 | 0.9910 | 0.9920 | 0.9950 | 0.9910 | 0.9930 |
| Pasture | 0.9913 | 0.9849 | 0.9799 | 0.9887 | 0.9850 | 0.9811 |
| PermanentCrop | 0.9811 | 0.9831 | 0.9704 | 0.9731 | 0.9677 | 0.9822 |
| Residential | 0.9958 | 0.9975 | 0.9967 | 0.9983 | 0.9958 | 0.9958 |
| River | 0.9910 | 0.9910 | 0.9900 | 0.9950 | 0.9940 | 0.9920 |
| SeaLake | 0.9975 | 0.9983 | 0.9983 | 1.0000 | 1.0000 | 0.9983 |

![Per-Class F1 Comparison](outputs/figures/comparison_per_class_f1.png)

> **Figure 5.** Per-class F1 comparison between ResNet-50 and ViT-Small/16 (mean of 3 runs). Horizontal bars show the F1 score for each land cover class.

Table 13 summarises the mean per-class F1 scores for each model.

**Table 13.** Mean per-class F1 scores (averaged over 3 runs).

| Class | ResNet-50 (mean) | ViT-Small/16 (mean) | Difference |
|:---|:---:|:---:|:---:|
| AnnualCrop | 0.9815 | 0.9756 | +0.0059 (ResNet) |
| Forest | 0.9967 | 0.9978 | +0.0011 (ViT) |
| HerbaceousVegetation | 0.9883 | 0.9858 | +0.0025 (ResNet) |
| Highway | 0.9880 | 0.9887 | +0.0007 (ViT) |
| Industrial | 0.9913 | 0.9930 | +0.0017 (ViT) |
| Pasture | 0.9854 | 0.9849 | +0.0005 (ResNet) |
| PermanentCrop | 0.9782 | 0.9743 | +0.0039 (ResNet) |
| Residential | 0.9967 | 0.9966 | +0.0001 (ResNet) |
| River | 0.9907 | 0.9937 | +0.0030 (ViT) |
| SeaLake | 0.9980 | 0.9994 | +0.0014 (ViT) |

The per-class analysis reveals a clear and interpretable pattern that aligns with the architectural hypotheses outlined in Section 3.1.7.

**ViT-Small/16 advantages.** ViT-Small/16 outperforms ResNet-50 on five classes: Forest, Highway, Industrial, River, and SeaLake. Notably, ViT achieves a perfect F1 score of 1.0000 on SeaLake in two out of three runs. These classes share a common characteristic: they tend to have relatively homogeneous spatial structures or benefit from global contextual understanding. For instance, SeaLake and River are water bodies with uniform spectral signatures, and Industrial areas often have distinctive large-scale spatial patterns (buildings, parking lots, infrastructure) that are better captured by global self-attention. The ViT advantage on River (+0.0030) is particularly noteworthy, as river classification requires understanding the elongated, connected spatial structure that spans large portions of the image—a scenario where global attention excels.

**ResNet-50 advantages.** ResNet-50 outperforms ViT-Small/16 on five classes: AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop, and Residential. These are predominantly vegetation-related classes characterised by fine-grained local textures that vary at small spatial scales. The largest ResNet advantages occur for AnnualCrop (+0.0059) and PermanentCrop (+0.0039), both of which are crop types distinguished primarily by subtle texture differences in leaf patterns, row spacing, and canopy structure—local features that are naturally captured by hierarchical convolutions. The relatively strong local inductive biases of CNNs appear to provide an advantage for discriminating between visually similar vegetation types.

**Challenging classes.** Both models find PermanentCrop and AnnualCrop the most challenging classes (lowest F1 scores across all runs), reflecting the inherent difficulty of distinguishing between crop types from satellite imagery alone. This inter-class confusion is further analysed in the confusion matrix analysis below.

#### 3.3.3 Confusion Matrix Analysis

Figures 6 and 7 present the row-normalised confusion matrices for the best-performing runs of each model (ResNet-50 seed 0 and ViT-Small/16 seed 0, respectively).

![ResNet-50 Confusion Matrix (Seed 0)](outputs/figures/resnet50_seed0_confusion.png)

> **Figure 6.** Row-normalised confusion matrix for ResNet-50 (seed 0, Macro-F1 = 0.9909). Diagonal entries represent correct classification rates (recall) for each class.

![ViT-Small/16 Confusion Matrix (Seed 0)](outputs/figures/vit_small_seed0_confusion.png)

> **Figure 7.** Row-normalised confusion matrix for ViT-Small/16 (seed 0, Macro-F1 = 0.9896). Same layout as Figure 6.

Several systematic confusion patterns are observed in both architectures:

**AnnualCrop ↔ PermanentCrop.** The most prominent off-diagonal entries in both confusion matrices involve mutual confusion between AnnualCrop and PermanentCrop. This is expected, as both classes represent agricultural land with similar spectral signatures—particularly in the visible and NIR bands—and are distinguished primarily by temporal patterns (annual vs. perennial growth cycles) that are not captured in single-date imagery.

**HerbaceousVegetation ↔ Pasture.** Both models also show some confusion between HerbaceousVegetation and Pasture, which is understandable given that both are grassland-type land covers with similar spectral characteristics. The distinction between managed grazing land (Pasture) and natural grassland (HerbaceousVegetation) requires subtle contextual cues such as field boundaries, grazing patterns, and surrounding land use.

**Architecture-specific differences.** Comparing the two confusion matrices reveals that ViT-Small/16 achieves marginally higher recall on water-related classes (River, SeaLake) and structural classes (Industrial), while ResNet-50 achieves marginally higher recall on vegetation classes (AnnualCrop, PermanentCrop, HerbaceousVegetation). This pattern is consistent with the per-class F1 analysis and supports the hypothesis that each architecture's inductive biases provide complementary advantages for different land cover types.

#### 3.3.4 Computational Efficiency

Table 14 compares the computational characteristics of both models.

**Table 14.** Computational efficiency comparison.

| Metric | ResNet-50 | ViT-Small/16 |
|:---|:---:|:---:|
| Total parameters | 23.56 M | 22.65 M |
| Inference speed | 15.54 ms/img | 15.72 ms/img |
| Training time (seed 0) | 61.1 min | 99.0 min |
| Training time (seed 1) | 59.7 min | 61.2 min |
| Training time (seed 2) | 60.5 min | 60.4 min |
| Mean training time | 60.4 min | 73.5 min |

Both models have similar parameter counts, with ViT-Small/16 being slightly smaller (22.65M vs. 23.56M). Inference speeds are nearly identical (15.54 vs. 15.72 ms/img), indicating that neither architecture has a significant efficiency advantage during deployment.

Training times are generally comparable, with both models completing a full 10-epoch run in approximately 60 minutes. The seed 0 run of ViT-Small/16 is an outlier at 99.0 minutes, likely due to system-level factors (e.g., GPU thermal throttling, background processes) rather than inherent algorithmic differences, as seeds 1 and 2 show training times almost identical to ResNet-50.

From a practical deployment perspective, both models are equally suitable for operational remote sensing classification tasks, with comparable computational requirements for both training and inference.

---

## 4. Conclusion and Future Work

### 4.1 Conclusion

This study has presented a rigorous, controlled comparison of two fundamentally different deep learning architectures—ResNet-50 (CNN) and ViT-Small/16 (Transformer)—for land cover classification on multispectral Sentinel-2 satellite imagery. By maintaining strictly identical experimental conditions—including the same dataset split, data preprocessing, augmentation strategy, optimizer, learning rate schedule, and evaluation metrics—this work ensures that observed performance differences are attributable solely to architectural design rather than experimental configuration.

The key findings of this study are as follows:

1. **Equivalent overall performance.** Both architectures achieve excellent classification accuracy on the EuroSAT dataset, with ResNet-50 attaining an overall accuracy of 98.98 ± 0.13% and ViT-Small/16 achieving 98.93 ± 0.08%. The difference of 0.05 percentage points is well within one standard deviation and is not statistically significant. This demonstrates that, under fair conditions with appropriate transfer learning, CNN and Transformer architectures are equally capable of high-accuracy multispectral remote sensing classification.

2. **Complementary per-class strengths.** Detailed per-class analysis reveals that each architecture has distinctive advantages aligned with its inductive biases. ResNet-50, with its hierarchical convolutions and local receptive fields, performs better on texture-rich vegetation classes (AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop). ViT-Small/16, with its global self-attention mechanism, excels at structurally homogeneous or context-dependent classes (Industrial, River, SeaLake). This complementarity suggests that the choice of architecture should be informed by the target application: vegetation-focused studies may benefit from CNNs, while urban or hydrological mapping may benefit from Transformers.

3. **Greater stability of ViT.** ViT-Small/16 exhibits notably lower variance across independent runs (OA std = 0.0008 vs. 0.0013 for ResNet-50), indicating greater robustness to random initialisation and training order. This stability may be valuable in operational settings where consistent, reproducible results are critical.

4. **Effectiveness of physics-aware band expansion.** The proposed physics-aware band expansion strategy successfully adapts 3-channel ImageNet pre-trained models to 13-band multispectral input, enabling both architectures to achieve near-99% accuracy. This approach preserves the benefits of transfer learning while fully utilising all available spectral information, offering a principled alternative to band discarding or naive replication.

5. **Comparable computational costs.** Both models have similar parameter counts (~23M) and inference speeds (~15.6 ms/img), indicating that neither architecture imposes a significant computational penalty for deployment.

In summary, this study demonstrates that the CNN-versus-Transformer debate does not have a single winner in the context of multispectral remote sensing classification. Rather, both paradigms are highly effective and offer complementary strengths. The choice between them should be guided by the specific requirements of the application, including the target land cover classes, the importance of result stability, and the available computational resources.

### 4.2 Future Work

Several directions for future research are identified based on the findings and limitations of this study:

1. **Extended training duration.** The current experiments use 10 training epochs. Future work should investigate the effect of training for 50 or more epochs to determine whether longer training reveals more pronounced differences between CNN and Transformer convergence behaviour, and whether one architecture benefits disproportionately from additional training.

2. **Hybrid architectures.** Recent architectures such as Swin Transformer and ConvNeXt combine elements of both CNN and Transformer designs. Evaluating these hybrid models under the same controlled conditions would provide a more complete picture of the architectural landscape for remote sensing classification.

3. **Multi-dataset evaluation.** The current study is limited to the EuroSAT dataset. Extending the comparison to other remote sensing benchmarks (e.g., UC Merced, AID, NWPU-RESISC45, BigEarthNet) would test the generalisability of the findings across different spatial resolutions, geographic regions, and class taxonomies.

4. **Alternative band expansion strategies.** The physics-aware mapping is one of several possible strategies for adapting pre-trained models to multispectral input. Future work could explore learned band projections, principal component analysis (PCA) based reduction, or attention-based band weighting mechanisms, and compare their effectiveness against the physics-aware approach.

5. **Attention visualisation and interpretability.** ViT's attention maps provide a natural mechanism for visualising which spatial regions the model focuses on for each class. Systematic analysis of attention patterns could yield deeper insights into why ViT excels at certain classes and inform the design of more interpretable remote sensing classifiers.

6. **Multi-temporal analysis.** The current study uses single-date imagery. Incorporating multi-temporal image sequences could address the AnnualCrop/PermanentCrop confusion identified in this study, as temporal growth patterns provide a strong discriminative signal between annual and perennial crops.

7. **Semi-supervised and self-supervised pre-training.** Exploring domain-specific pre-training on unlabelled satellite imagery (e.g., using contrastive learning or masked autoencoders) could further improve performance, particularly for Transformer architectures that are known to benefit from large-scale pre-training.

8. **Model compression.** Investigating knowledge distillation, pruning, and quantisation techniques could reduce the computational footprint of both models, enabling deployment on edge devices for real-time satellite image analysis.

---

## References

[1] Mountrakis, G., Im, J. and Ogole, C. (2011) "Support vector machines in remote sensing: A review." *ISPRS Journal of Photogrammetry and Remote Sensing*, 66(3), pp.247-259.

[2] Deng, J. et al. (2009) "ImageNet: A large-scale hierarchical image database." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp.248-255.

[3] Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012) "ImageNet classification with deep convolutional neural networks." *Advances in Neural Information Processing Systems*, 25.

[4] Castelluccio, M. et al. (2015) "Land use classification in remote sensing images by convolutional neural networks." *arXiv preprint arXiv:1508.00092*.

[5] Zhu, X.X. et al. (2017) "Deep learning in remote sensing: A comprehensive review and list of resources." *IEEE Geoscience and Remote Sensing Magazine*, 5(4), pp.8-36.

[6] He, K. et al. (2016) "Deep residual learning for image recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp.770-778.

[7] Vaswani, A. et al. (2017) "Attention is all you need." *Advances in Neural Information Processing Systems*, 30.

[8] Dosovitskiy, A. et al. (2020) "An image is worth 16x16 words: Transformers for image recognition at scale." *arXiv preprint arXiv:2010.11929*.

[9] Touvron, H. et al. (2021) "Training data-efficient image transformers & distillation through attention." *Proceedings of the International Conference on Machine Learning (ICML)*, pp.10347-10357.

[10] Bazi, Y. et al. (2021) "Vision transformers for remote sensing image classification." *Remote Sensing*, 13(3), p.516.

[11] He, X., Chen, Y. and Lin, Z. (2021) "Spatial-spectral transformer for hyperspectral image classification." *Remote Sensing*, 13(3), p.498.

[12] Aleissaee, A.A. et al. (2023) "Transformers in remote sensing: A survey." *Remote Sensing*, 15(7), p.1860.

[13] Helber, P. et al. (2019) "EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), pp.2217-2226.

[14] Drusch, M. et al. (2012) "Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services." *Remote Sensing of Environment*, 120, pp.25-36.

[15] Neumann, M. et al. (2019) "In-domain representation learning for remote sensing." *arXiv preprint arXiv:1911.06721*.

[16] Maurício, J., Domingues, I. and Bernardino, J. (2023) "Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review." *Applied Sciences*, 13(9), p.5521.

[17] Loshchilov, I. and Hutter, F. (2019) "Decoupled weight decay regularization." *Proceedings of the International Conference on Learning Representations (ICLR)*.

[18] Wightman, R. (2019) "PyTorch Image Models (timm)." GitHub repository. Available at: https://github.com/rwightman/pytorch-image-models.
