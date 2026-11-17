---
title: DEIMv2 深度解析：DINOv3 特征如何适配实时 DETR
description: 从最新论文、官方源码、模型配置和 Hugging Face 权重出发，拆解 DEIMv2 的双 Backbone 路线、STA、Encoder、Decoder、Dense O2O、Copy-Blend、损失函数及部署流程。
---

# DEIMv2 深度解析：DINOv3 特征如何适配实时 DETR

> 本文重点不是复述论文摘要，而是回答三个问题：DEIMv2 的网络究竟由哪些模块组成？论文提出的设计在官方代码中是如何实现的？如果要在自己的数据集上训练和部署，应该从哪里开始？

**资料口径：** 论文采用目前可获取的 v5 版本（2026-10-27）；代码以作者官方 GitHub 仓库当前公开分支及其配置为准；模型权重和 Hub 配置以作者的 Intellindust Hugging Face 组织为准。论文、源码和模型卡存在差异的地方会单独标注。

## 一、先理解 DEIMv2 到底是什么

DEIMv2 的论文名称是 **Real-Time Object Detection Meets DINOv3**，作者为 Shihua Huang、Yongjie Hou、Longfei Liu、Xuanlong Yu 和 Xi Shen。论文最初发表于 2025 年 9 月，后续持续修订；本文使用 v5 版本。

DEIMv2 不是简单地把 DINOv3 接到一个 DETR 检测头上。它是一套完整的实时检测方案，主要由四部分组成：

1. **两条 Backbone 路线：** S、M、L、X 使用 DINOv3 或由 DINOv3 蒸馏得到的 ViT；Nano、Pico、Femto、Atto 使用裁剪后的 HGNetv2。
2. **适配不同 Backbone 的特征融合：** DINOv3 路线使用 Spatial Tuning Adapter（STA）产生多尺度特征；轻量 HGNetv2 路线根据输出尺度选择 HybridEncoder 或 LiteEncoder。
3. **更高效的 Transformer Decoder：** 使用 Deformable Cross-Attention、SwiGLU FFN、RMSNorm 和共享 Query Position Embedding，并保留 D-FINE 风格的细粒度边界框分布细化。
4. **改进训练监督：** 继承 DEIM 的 Dense O2O 和 MAL，进一步加入对象级 Copy-Blend。

可以先用一句话概括：

**DEIMv2 = RT-DETR 风格的整体检测框架 + DINOv3/STA 或轻量 HGNetv2 Backbone + DEIM 训练策略 + D-FINE 风格的定位细化。**

### 1.1 论文与项目入口

| 资料 | 用途 |
|---|---|
| [DEIMv2 论文 v5 PDF](https://arxiv.org/pdf/2509.20787v5) | 方法、实验、消融与最新论文表格 |
| [DEIMv2 arXiv 页面](https://arxiv.org/abs/2509.20787) | 论文版本与摘要 |
| [DEIMv2 官方 GitHub](https://github.com/Intellindust-AI-Lab/DEIMv2) | README、模型配置、训练、推理和部署代码 |
| [DEIMv2 官方项目页面](https://intellindust-ai-lab.github.io/projects/DEIMv2/) | 项目介绍 |
| [Intellindust Hugging Face 组织](https://huggingface.co/Intellindust) | 作者发布的模型权重 |
| [Hugging Face Transformers：DEIMv2](https://huggingface.co/docs/transformers/model_doc/deimv2) | Transformers 中的模型接口，注意与作者原始 Hub 权重区分 |

## 二、DEIM、RT-DETR、D-FINE、DINOv3 分别负责什么

这几个名称经常一起出现，但它们并不是同一层面的模型。

| 方法 | 主要解决的问题 | 在 DEIMv2 中的关系 |
|---|---|---|
| RT-DETR | 实时端到端检测器的整体组织方式，包括 Backbone、Hybrid Encoder 和 Transformer Decoder | DEIMv2 的整体架构沿用这一组织方式 |
| DEIM | DETR 的匹配和训练效率，包括 Dense O2O 与 MAL | DEIMv2 继承并增强其训练策略 |
| D-FINE | 边界框回归的细粒度分布细化，以及 FGL、DDF 等局部定位损失 | DEIMv2 的 Decoder 和定位损失采用相关设计 |
| DINOv3 | 自监督预训练得到的视觉特征 Backbone | 为 S、M、L、X 提供语义特征 |
| HGNetv2 | 面向实时检测的高效 CNN Backbone | 为 Nano、Pico、Femto、Atto 提供轻量特征 |

### 2.1 DEIM 与 DEIMv2 的区别

DEIM 的核心贡献主要在训练层面：

- Dense O2O：通过数据增强增加单张训练图像中的有效目标数量，让 O2O 匹配获得更丰富的监督。
- MAL：让分类监督感知预测框与真实框之间的匹配质量。

DEIMv2 保留这套思路，但进一步扩展了模型结构：

- 为较大模型引入 DINOv3 和 STA。
- 为极轻量模型设计 HGNetv2 裁剪路线。
- 简化 Decoder 中的 FFN 和归一化计算。
- 在对象级别增加 Copy-Blend。
- 使用 D-FINE 风格的分布式定位细化。

所以，DEIMv2 不只是一个训练技巧，也不只是一个 Backbone。它同时改进了模型结构与训练过程。

### 2.2 DEIMv2 与 RT-DETR、D-FINE 的关系

从官方模型的实际前向流程看，DEIMv2 仍然是典型的端到端 DETR 风格检测器：

- Backbone 提取特征。
- Encoder 融合多尺度特征，并生成初始候选。
- Decoder 使用 Object Queries 迭代细化检测结果。
- PostProcessor 选择最终预测。

它不是 YOLO 式的密集网格检测头，也不是直接把 D-FINE 换一个名字。

更准确地说，DEIMv2 沿用 RT-DETR 的整体架构组织方式，在 Decoder 和分布式定位部分吸收 D-FINE 的设计，同时将 DEIM 的匹配和训练策略融入完整检测器。

## 三、整体网络架构

DEIMv2 的模型前向路径可以简化为：

\[
\text{Image}
\rightarrow
\text{Backbone}
\rightarrow
\text{Encoder}
\rightarrow
\text{Decoder}
\rightarrow
\text{PostProcessor}
\]

其中最重要的结构差异是 Backbone 路线不同，导致 Encoder 也分成两种实现。

### 3.1 两条模型路线

~~~mermaid
flowchart TD
    A["输入图像"] --> B{"模型规格"}
    B -->|"S / M / L / X"| C["DINOv3 或蒸馏 ViT-Tiny"]
    C --> D["STA：语义特征 + 空间先验"]
    D --> E["三个尺度：1/8、1/16、1/32"]
    E --> F["HybridEncoder"]
    B -->|"Nano"| G["HGNetv2-B0"]
    G --> H["两个尺度：1/16、1/32"]
    H --> I["HybridEncoder"]
    B -->|"Pico / Femto / Atto"| J["裁剪 HGNetv2"]
    J --> K["1/16 特征"]
    K --> L["LiteEncoder"]
    L --> M["两个尺度：1/16、1/32"]
    F --> N["DEIM Transformer Decoder"]
    I --> N
    M --> N
    N --> O["分类与边界框预测"]
    O --> P["Top-K 后处理"]
    P --> Q["检测结果"]
~~~

### 3.2 八种模型规格

| 规格 | Backbone | Encoder | 特征尺度数 | Decoder 层数 | Queries | 输入尺寸 |
|---|---|---|---:|---:|---:|---:|
| Atto | HGNetv2-Atto | LiteEncoder | 2 | 3 | 100 | 320 |
| Femto | HGNetv2-Femto | LiteEncoder | 2 | 3 | 150 | 416 |
| Pico | HGNetv2-Pico | LiteEncoder | 2 | 3 | 200 | 640 |
| Nano（N） | HGNetv2-B0 | HybridEncoder | 2 | 3 | 300 | 640 |
| S | ViT-Tiny | HybridEncoder | 3 | 4 | 300 | 640 |
| M | ViT-Tiny+ | HybridEncoder | 3 | 4 | 300 | 640 |
| L | DINOv3 ViT-Small | HybridEncoder | 3 | 4 | 300 | 640 |
| X | DINOv3 ViT-Small+ | HybridEncoder | 3 | 6 | 300 | 640 |

这里的特征尺度数是送入 Decoder 的 Feature Levels 数量，不等同于 Backbone 自身的 Stage 数量。

### 3.3 官方代码中的主干前向路径

官方的 DEIM 模型类在前向过程中基本只做三件事：

~~~python
def forward(self, x, targets=None):
    x = self.backbone(x)
    x = self.encoder(x)
    x = self.decoder(x, targets)
    return x
~~~

这段代码说明：

- Backbone 负责产生特征。
- Encoder 负责特征交互与多尺度融合。
- Decoder 负责候选选择和逐层预测。

训练阶段，Decoder 还会返回中间层辅助输出、Encoder 辅助输出以及去噪分支输出，供 Criterion 计算损失。推理阶段则主要保留最终预测。

源码：[engine/deim/deim.py](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim.py)。

## 四、DINOv3 Backbone 与 STA

这是 DEIMv2 最值得深入理解的部分。

### 4.1 为什么 DINOv3 需要额外的特征适配？

DINOv3 的 ViT Backbone 通过 Patch Embedding 将图像切分为 Patch Token，再经过 Transformer Block 提取特征。

以 Patch Size = 16、输入尺寸 640 × 640 为例：

\[
H_p=W_p=\frac{640}{16}=40
\]

因此，Patch Token 对应的空间网格为：

\[
40\times40
\]

不考虑 Class Token 时，特征可以写成：

\[
F_{\text{ViT}}
\in
\mathbb{R}^{B\times D\times40\times40}
\]

其中：

- \(B\)：Batch Size。
- \(D\)：ViT 的隐藏维度。
- \(40\times40\)：Patch 特征图的空间尺寸。

这相当于步长为 16 的单一空间尺度。

但目标检测中的目标尺寸差异很大：

- 小目标需要更高分辨率的特征。
- 中等目标需要中间尺度。
- 大目标需要更强的语义信息和较大的感受野。

因此，检测器通常需要类似 P3、P4、P5 的多尺度特征：

| 特征 | 步长 | 640 输入时的空间尺寸 |
|---|---:|---:|
| P3 | 8 | 80 × 80 |
| P4 | 16 | 40 × 40 |
| P5 | 32 | 20 × 20 |

如果直接使用 DINOv3 最后一层的 40 × 40 特征，就不能充分利用检测任务所需要的多尺度信息。

### 4.2 DINOv3 的中间层特征

DEIMv2 不只取 ViT 最后一层，而是从多个 Transformer Block 提取特征。

论文中举例使用第 5、8、11 层。因为 ViT 各层的 Patch Token 空间尺寸相同，都是 1/16 尺度，所以需要再通过 Resize 构造不同分辨率：

\[
\begin{aligned}
F_{1/8} &= \operatorname{Resize}(F_a,80,80)\\
F_{1/16} &= \operatorname{Resize}(F_b,40,40)\\
F_{1/32} &= \operatorname{Resize}(F_c,20,20)
\end{aligned}
\]

这里的 Resize 使用双线性插值。

需要注意：

**双线性插值本身没有可学习参数。** 它只是调整特征图空间尺寸，不会凭空生成新的语义信息。

多层特征提供不同深度的语义表达，插值则让它们拥有检测器需要的空间分辨率。

### 4.3 为什么还需要 Spatial Prior Module？

只从 ViT 中间层进行插值，虽然能得到不同分辨率的特征图，但这些特征仍然来自同一 Patch 尺度。

插值不会恢复原始图像中已经丢失的局部细节。

因此，DEIMv2 在 DINOv3 旁边增加一条轻量 CNN 分支，从原始图像中提取局部空间信息。

官方代码中的模块名是 **SpatialPriorModulev2**。

其结构为：

~~~mermaid
flowchart TD
    A["输入图像 3×H×W"] --> B["3×3 Conv，stride=2"]
    B --> C["SyncBatchNorm + GELU"]
    C --> D["3×3 MaxPool，stride=2"]
    D --> E["1/4 特征"]
    E --> F["Conv2，stride=2"]
    F --> G["1/8 特征"]
    G --> H["Conv3，stride=2"]
    H --> I["1/16 特征"]
    I --> J["Conv4，stride=2"]
    J --> K["1/32 特征"]
~~~

以输入 640 × 640、默认 inplanes=16 为例：

| 阶段 | 空间尺寸 | 通道数 |
|---|---:|---:|
| Stem | 160 × 160 | 16 |
| Conv2 | 80 × 80 | 32 |
| Conv3 | 40 × 40 | 64 |
| Conv4 | 20 × 20 | 64 |

可以看到，这条分支直接从图像中提取三个尺度的空间特征。

它的感受野相对较小，更偏向局部纹理、边缘和细节，而 DINOv3 提供较强的高层语义信息。

### 4.4 Bi-Fusion：语义特征和空间特征如何融合？

STA 的关键不只是增加一个 CNN 分支，而是把 CNN 特征与 DINOv3 特征融合。

对每个尺度，先将 DINOv3 特征调整到目标分辨率，再与空间先验特征在通道维度拼接：

\[
F_i^{\text{cat}}
=
\operatorname{Concat}
\left(
F_i^{\text{ViT}},
F_i^{\text{Spatial}}
\right)
\]

然后使用 1 × 1 Conv 进行通道投影：

\[
F_i^{\text{out}}
=
\operatorname{SyncBN}
\left(
\operatorname{Conv}_{1\times1}
(F_i^{\text{cat}})
\right)
\]

整体流程：

~~~mermaid
flowchart TD
    A["DINOv3 ViT"] --> B["提取三个中间层"]
    B --> C["Token 转二维特征图"]
    C --> D["Bilinear Resize"]
    D --> E["语义特征：1/8、1/16、1/32"]
    F["原始图像"] --> G["SpatialPriorModulev2"]
    G --> H["卷积下采样"]
    H --> I["细节特征：1/8、1/16、1/32"]
    E --> J["Concat"]
    I --> J
    J --> K["1×1 Conv"]
    K --> L["SyncBatchNorm"]
    L --> M["输出多尺度特征"]
~~~

这里要区分两个概念：

- **Bilinear Resize：** 参数无关，只改变特征图尺寸。
- **Bi-Fusion：** 包含 Concat、1 × 1 Conv 和归一化，存在可学习参数。

所以，不能把整个 STA 理解成一个完全无参数模块。论文强调的 parameter-free 主要是 ViT 单尺度特征转换成多尺度时使用的双线性插值。

### 4.5 S、M、L、X 的 Backbone 具体有什么区别？

| 规格 | ViT Backbone | Backbone 隐藏维度 | 预训练方式 |
|---|---|---:|---|
| S | ViT-Tiny | 192 | 从 DINOv3 ViT-Small 蒸馏 |
| M | ViT-Tiny+ | 256 | 从 DINOv3 ViT-Small 蒸馏 |
| L | DINOv3 ViT-Small | 384 | 官方 DINOv3 预训练权重 |
| X | DINOv3 ViT-Small+ | 384 | 官方 DINOv3 预训练权重 |

四种 Backbone 均采用 12 层 ViT。

S、M 的 Tiny 和 Tiny+ 是作者为了实时检测设计的紧凑 Backbone，不应与 Meta 官方发布的标准 DINOv3 模型规格混为一谈。

L、X 则直接使用 DINOv3 的 ViT-Small 和 ViT-Small+。

### 4.6 STA 输出通道并不一定等于 ViT 隐藏维度

在官方代码中，DINOv3 原始特征与空间先验特征拼接后，会经过 1 × 1 Conv 投影到检测网络需要的通道数。

例如：

- S：ViT 隐藏维度 192，最终输出通道 192。
- M：ViT 隐藏维度 256，最终输出通道 256。
- L：ViT 原始隐藏维度 384，当前官方 YAML 将 STA 输出投影到 224。
- X：ViT 原始隐藏维度 384，最终投影到 256。

这意味着，DINOv3 原始 Backbone 的维度、STA 输出维度、Encoder 隐藏维度和 Decoder 隐藏维度是不同概念。

不要看到 ViT-Small 的隐藏维度是 384，就认为后面的检测网络也必须全部使用 384 通道。

### 4.7 配置文件中的中间层索引存在差异

这里有一个需要特别注意的源码问题。

目前公开资料中的索引并不完全一致：

| 资料 | S / M 使用的 interaction_indexes |
|---|---|
| 官方 S / M YAML | [3, 7, 11] |
| 官方 hf_models.ipynb 示例 | [5, 8, 11] |
| 作者发布的 S 模型 config.json | [5, 8, 11] |
| 论文结构图的示例 | 第 5、8、11 层 |

官方 Tiny Backbone 的实现会遍历 Transformer Blocks，并在索引命中 return_layers 时保存输出。因此，这些索引不是可以随意替换的注释信息。

目前公开的 S 模型 YAML 与 Hub 配置存在差异，无法仅凭公开文件确认 Hub checkpoint 实际训练时使用的索引版本。复现训练时应固定仓库版本、YAML 和 checkpoint，不要混用不同来源的配置。

L、X 的官方 YAML 使用 [5, 8, 11]。

源码：

- [DINOv3 STA Adapter](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/dinov3_adapter.py)
- [ViT-Tiny 实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/vit_tiny.py)

## 五、HGNetv2 轻量化路线

DINOv3 路线的语义特征很强，但 ViT 的计算和显存访问并不适合所有极低算力场景。

因此，DEIMv2 为 Nano、Pico、Femto、Atto 设计了另一条路线：以 HGNetv2-B0 为基础，逐步裁剪深度和通道宽度。

### 5.1 四种轻量 Backbone 如何裁剪？

论文给出的裁剪思路：

1. **Nano：** 使用 HGNetv2-B0。
2. **Pico：** 从 HGNetv2-B0 移除第四阶段，只保留到 1/16 尺度。
3. **Femto：** 在 Pico 的基础上，将最后阶段的 Block 数量从 2 个减少到 1 个。
4. **Atto：** 在 Femto 的基础上，将最后一个 Block 的通道数从 512 缩减到 256。

这是一条逐步降低计算量的路线，而不是简单地把同一个网络的宽度系数连续调小。

### 5.2 Nano 为什么仍然使用 HybridEncoder？

Nano 使用 HGNetv2-B0，官方配置输出两个尺度：

- 1/16：512 通道。
- 1/32：1024 通道。

因此，Nano 可以直接使用两尺度 HybridEncoder。

它的主要配置为：

| 参数 | Nano |
|---|---:|
| Backbone | HGNetv2-B0 |
| Backbone 输出通道 | [512, 1024] |
| Feature Strides | [16, 32] |
| Encoder Hidden Dimension | 128 |
| Decoder Hidden Dimension | 128 |
| Decoder Layers | 3 |
| Queries | 300 |

### 5.3 Pico、Femto、Atto 为什么使用 LiteEncoder？

Pico、Femto、Atto 的 Backbone 主要输出单个 1/16 尺度特征。

如果直接使用完整的多尺度 HybridEncoder，就需要额外计算和结构。

因此官方设计了 LiteEncoder：从一个 1/16 特征开始，轻量地生成 1/32 特征，然后完成两尺度融合。

| 参数 | Pico | Femto | Atto |
|---|---:|---:|---:|
| Backbone 输出通道 | 512 | 512 | 256 |
| Backbone 输出步长 | 16 | 16 | 16 |
| Encoder Hidden Dimension | 112 | 96 | 64 |
| Decoder Hidden Dimension | 112 | 96 | 64 |
| Decoder Layers | 3 | 3 | 3 |
| Queries | 200 | 150 | 100 |

### 5.4 LiteEncoder 的数据流

LiteEncoder 的结构可以分成四步：

1. 使用 1 × 1 Conv 将输入通道投影到统一维度。
2. 使用 AvgPool 和卷积下采样，构造 1/32 特征。
3. 使用 GAP Fusion 融合全局信息。
4. 通过 FPN/PAN 完成两尺度特征交互。

~~~mermaid
flowchart TD
    A["HGNetv2 输出 1/16 特征"] --> B["1×1 Conv + BN"]
    B --> C["AvgPool 下采样"]
    C --> D["1/32 特征"]
    D --> E["GAP Fusion"]
    B --> F["1/16 特征"]
    E --> G["上采样并融合"]
    F --> G
    G --> H["FPN：RepNCSPELAN"]
    H --> I["下采样"]
    I --> J["与 1/32 特征融合"]
    J --> K["PAN：RepNCSPELAN"]
    K --> L["输出 1/16、1/32"]
~~~

### 5.5 GAP Fusion 是什么？

官方代码中的 GAP_Fusion 逻辑很简单：

\[
G=\operatorname{GAP}(F)
\]

\[
F'=F+G
\]

其中：

- \(F\)：输入特征图。
- \(G\)：全局平均池化后得到的 1 × 1 特征。
- \(F'\)：融合全局上下文后的特征。

由于空间维度为 1 × 1，执行加法时会广播到原特征图的所有空间位置。

最后再经过 1 × 1 Conv、归一化和激活。

GAP Fusion 的作用是将全局信息注入局部特征，而不是增加新的检测尺度。

源码：[engine/deim/lite_encoder.py](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/lite_encoder.py)。

## 六、HybridEncoder 的内部结构

HybridEncoder 是 RT-DETR 系列中非常重要的模块。DEIMv2 保留了它的核心思想，但根据模型路线调整了通道、融合方式和模块实现。

### 6.1 输入通道统一

不同 Backbone 输出的通道数可能不同。

HybridEncoder 首先使用 1 × 1 Conv 和 BatchNorm，将输入特征投影到统一的 Hidden Dimension。

例如 DEIMv2-S：

\[
[P_3,P_4,P_5]
\]

输入通道：

\[
[192,192,192]
\]

Encoder Hidden Dimension：

\[
D=192
\]

由于输入通道已经一致，可以使用 Identity，不必额外投影。

而 Nano 的输入通道为 [512,1024]，需要投影到 [128,128]。

### 6.2 Transformer Encoder 只处理指定尺度

HybridEncoder 并不是把所有尺度都送进 Transformer。

官方配置中：

- DINOv3 路线使用 use_encoder_idx = [2]。
- Nano 使用 use_encoder_idx = [1]。

也就是只对最高层、空间分辨率最低的特征做 Transformer 编码。

以 S 为例：

- P3：80 × 80。
- P4：40 × 40。
- P5：20 × 20。

只有 P5 被送入 Transformer Encoder。

首先将二维特征展开：

\[
F\in\mathbb{R}^{B\times C\times H\times W}
\]

转换为：

\[
F_{\text{flat}}
\in
\mathbb{R}^{B\times HW\times C}
\]

随后加入 2D Sine-Cosine Position Embedding，进行 Transformer 编码，再恢复为二维特征图。

### 6.3 FPN：自顶向下融合

FPN 从高层语义特征开始：

1. 将 P5 通过横向卷积处理。
2. 上采样到 P4 尺寸。
3. 与 P4 融合。
4. 再上采样到 P3 尺寸。
5. 与 P3 融合。

DEIMv2 的相关配置使用 fuse_op = sum，因此融合是逐元素相加：

\[
F_{\text{fuse}}=
F_{\text{upsample}}+F_{\text{lateral}}
\]

这要求两个特征具有相同的空间尺寸和通道数。

### 6.4 PAN：自底向上融合

PAN 再从高分辨率特征向下融合：

1. 对 P3 进行下采样。
2. 与 P4 融合。
3. 再对结果下采样。
4. 与 P5 融合。

这样同时实现高层语义向低层传递、低层细节向高层传递。

~~~mermaid
flowchart TD
    A["P3：1/8"] --> B["横向特征"]
    C["P4：1/16"] --> D["横向特征"]
    E["P5：1/32"] --> F["Transformer Encoder"]
    F --> G["上采样"]
    G --> H["与 P4 相加"]
    H --> I["FPN Block"]
    I --> J["上采样"]
    J --> K["与 P3 相加"]
    K --> L["FPN 输出"]
    L --> M["下采样"]
    M --> N["与中层特征相加"]
    N --> O["PAN Block"]
    O --> P["下采样"]
    P --> Q["与高层特征相加"]
    Q --> R["PAN 输出"]
~~~

这里的结构图表达的是 HybridEncoder 的信息流，不代表每个卷积子模块的逐层精确结构。

### 6.5 HybridEncoder 与 LiteEncoder 的区别

| 对比项 | HybridEncoder | LiteEncoder |
|---|---|---|
| 输入 | 通常为 2 或 3 个尺度 | 通常为单个 1/16 尺度 |
| Transformer | 支持指定尺度进行 Transformer 编码 | 不使用该 Transformer Encoder 路径 |
| 多尺度构建 | 使用已有 Backbone 多尺度特征 | 从单尺度特征生成第二尺度 |
| 融合方式 | FPN + PAN | GAP Fusion + FPN + PAN |
| 主要使用模型 | S、M、L、X、Nano | Pico、Femto、Atto |

## 七、DEIM Transformer Decoder

Encoder 输出特征后，Decoder 负责生成最终检测结果。

DEIMv2 的 Decoder 不是普通的 Transformer Decoder，而是结合了 Deformable Attention 和 D-FINE 风格分布回归的结构。

### 7.1 Encoder 输出如何生成 Object Queries？

假设 S 模型输入为 640 × 640，Encoder 输出三个尺度：

- P3：80 × 80。
- P4：40 × 40。
- P5：20 × 20。

将三个特征图展平后，空间 Token 数量为：

\[
N=80^2+40^2+20^2
\]

\[
N=6400+1600+400=8400
\]

所以 Encoder Memory 可以表示为：

\[
M\in\mathbb{R}^{B\times8400\times192}
\]

接下来，Decoder 会为各个空间位置生成初始 Anchor，并通过 Encoder 分类分支得到初始分类分数。

随后按照分类分数选择 Top-K 候选：

\[
Q=\operatorname{TopK}(S_{\text{encoder}},K)
\]

S 模型中：

\[
K=300
\]

得到 300 个候选 Query。

这些候选包括：

- Query Content：候选位置对应的 Encoder 特征。
- Reference Boxes：候选位置对应的初始参考框。
- Query Position Embedding：由参考框位置编码得到。

默认配置下，Query Content 来自选中的 Encoder 特征，而不是一组完全独立学习的 Query Embedding。

### 7.2 一个 Decoder Layer 包含什么？

每层主要包含：

1. Self-Attention。
2. Multi-Scale Deformable Cross-Attention。
3. Gateway 或 Residual。
4. SwiGLU FFN。
5. RMSNorm。

~~~mermaid
flowchart TD
    A["Query Features"] --> B["Self-Attention"]
    B --> C["Residual + RMSNorm"]
    C --> D["Multi-Scale Deformable Cross-Attention"]
    E["Encoder Multi-scale Memory"] --> D
    D --> F{"use_gateway"}
    F -->|"True"| G["Gate：融合残差与 Cross-Attention"]
    F -->|"False"| H["Residual + RMSNorm"]
    G --> I["SwiGLU FFN"]
    H --> I
    I --> J["Residual + RMSNorm"]
    J --> K["更新 Query"]
    K --> L["分类与边界框预测"]
~~~

### 7.3 Self-Attention

Self-Attention 负责不同 Object Queries 之间的信息交互。

例如两个 Query 分别预测相邻的两株植物，它们之间可能存在空间关系和遮挡关系。

Self-Attention 可以让不同 Query 交换信息。

### 7.4 Multi-Scale Deformable Cross-Attention

传统 Cross-Attention 会让 Query 与大量 Key 进行交互。

DEIMv2 使用 Multi-Scale Deformable Attention，根据 Reference Points 在不同尺度特征图中采样少量相关位置。

这样不必让每个 Query 都与所有空间位置进行完整 Attention 计算。

对于密集场景，这种设计能够让 Query 聚焦于候选目标附近的特征，而不是均匀地处理所有空间位置。

### 7.5 Gateway

官方 Decoder 支持 use_gateway。

当开启 Gateway 时，模型将 Cross-Attention 输出和残差分支拼接，通过线性层生成两个 Gate：

\[
[g_1,g_2]=
\sigma(W[x_1,x_2]+b)
\]

然后融合：

\[
F_{\text{out}}=
\operatorname{RMSNorm}(g_1x_1+g_2x_2)
\]

其中：

- \(x_1\)：原 Query 特征。
- \(x_2\)：Cross-Attention 输出。
- \(g_1,g_2\)：两个可学习门控系数。

如果不使用 Gateway，则采用普通残差加法和 RMSNorm。

S、M、L、X 和 Nano 的配置默认使用 Gateway；Pico、Femto、Atto 的配置关闭 Gateway。

### 7.6 SwiGLU FFN

传统 FFN 通常是两层线性映射：

\[
\operatorname{FFN}(x)=
W_2\sigma(W_1x)
\]

DEIMv2 使用 SwiGLUFFN：

\[
\operatorname{SwiGLU}(x)=
\operatorname{SiLU}(xW_1)
\odot(xW_2)
\]

然后通过输出投影：

\[
y=\operatorname{SwiGLU}(x)W_3
\]

其中：

- \(W_1\)：Gate 分支。
- \(W_2\)：Value 分支。
- \(W_3\)：输出投影。
- \(\odot\)：逐元素乘法。

官方代码中使用 SiLU 作为门控激活函数。

### 7.7 RMSNorm

RMSNorm 根据输入特征的均方根进行缩放：

\[
\operatorname{RMSNorm}(x)=
\frac{x}
{\sqrt{\operatorname{Mean}(x^2)+\epsilon}}
\odot\gamma
\]

其中：

- \(\epsilon\)：数值稳定项，源码默认值为 \(10^{-6}\)。
- \(\gamma\)：可学习缩放参数。

与 LayerNorm 不同，RMSNorm 不需要先减去均值。

DEIMv2 在 Decoder 中使用 RMSNorm 替换常规 LayerNorm。

### 7.8 共享 Query Position Embedding

这是一个容易忽略但很重要的效率设计。

Decoder 需要根据 Reference Points 生成 Query Position Embedding：

\[
P_q=\operatorname{MLP}(R_q)
\]

传统实现可能在每一层重复生成或更新位置编码。

DEIMv2 的实现是在 Decoder 开始时根据初始 Reference Points 计算：

\[
P_q=\operatorname{MLP}(R_q^{(0)})
\]

然后在后续 Decoder Layer 中复用这一位置编码。

官方代码中，query_pos_embed 在 Decoder Layer 循环之前计算一次，再传给每一层。

这样可以减少重复的位置编码计算。

### 7.9 D-FINE 风格的边界框分布细化

DEIMv2 的边界框预测不是简单地直接回归四个连续坐标。

它保留了 D-FINE 的 Fine-grained Distribution Refinement 思路。

对于一个边界框，模型分别预测左、上、右、下四个方向的距离分布。

官方配置中常见：

- reg_max = 32。
- 每个方向有 33 个离散预测值。
- 四个方向分别预测。

假设某个方向的分类 Logits 为：

\[
z=[z_0,z_1,\ldots,z_{32}]
\]

先经过 Softmax：

\[
p_i=
\frac{\exp(z_i)}
{\sum_{j=0}^{32}\exp(z_j)}
\]

再与非均匀 Weighting Function 的值进行加权：

\[
d=\sum_{i=0}^{32}p_iW(i)
\]

其中 \(W(i)\) 不是简单的等间距整数索引，而是由 reg_max、up 和 reg_scale 共同决定的非均匀权重序列。

这也是为什么不能直接把 DEIMv2 的分布回归理解成普通的 0 到 32 均匀分箱。

得到四个方向的距离后，再根据 Reference Box 还原最终边界框。

这种设计可以让模型在离散分布上学习更细粒度的定位信息。

源码：

- [DEIM Decoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py)
- [D-FINE Decoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/dfine_decoder.py)
- [D-FINE Utils](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/dfine_utils.py)

## 八、Dense O2O、Copy-Blend 与匹配策略

DEIMv2 的另一个核心改进是训练过程。

理解这一部分，需要先明确：DEIMv2 仍然是 O2O 匹配检测器，并没有简单地把 Hungarian Matching 换成 YOLO 式的一对多分配。

### 8.1 为什么要 Dense O2O？

传统 DETR 使用 Hungarian Matching：

- 每个 Ground Truth 与一个预测框匹配。
- 没有匹配到 Ground Truth 的预测作为负样本。
- 匹配过程是一对一分配。

这种训练方式的一个问题是：单张训练图像中目标数量有限，参与正样本监督的目标数量也有限。

Dense O2O 的思路是通过数据增强增加每张训练图像中的有效目标数量。

例如，原本一张图只有 5 个目标，通过 Mosaic、MixUp 等方式，可以让训练图像中出现来自多个原始图像的目标。

这样仍然使用 O2O 匹配，但每张训练图像可以提供更多有效监督。

### 8.2 Mosaic 和 MixUp

Mosaic：

将多张训练图片组合为一张训练图片，同时更新各自的目标框和类别标签。

MixUp：

将两张图片按照一定比例混合：

\[
I=\beta I_1+(1-\beta)I_2
\]

目标标注也需要合并，并记录对应的混合权重。

这两种方法属于图像级增强。

### 8.3 Copy-Blend：对象级增强

DEIMv2 在此基础上提出 Copy-Blend。

论文的出发点是：不只是把多张图像组合起来，还可以从 Batch 中抽取对象，将它们加入其他训练图像。

流程：

1. 从当前 Batch 的标注中收集符合面积条件的目标。
2. 随机选择若干目标。
3. 根据目标框从源图像中裁剪 Patch。
4. 将 Patch 放置到目标图像的随机位置。
5. 通过 Blend 或 Copy 方式融合图像内容。
6. 同步增加对应的目标框、类别和面积标注。

~~~mermaid
flowchart TD
    A["当前 Batch"] --> B["收集所有目标框"]
    B --> C["根据面积阈值过滤"]
    C --> D["随机选择目标"]
    D --> E["从源图像裁剪 Patch"]
    E --> F["随机选择目标图像位置"]
    F --> G["Blend 或 Copy"]
    G --> H["更新目标图像"]
    H --> I["追加 Bounding Boxes 和 Labels"]
    I --> J["送入检测网络"]
~~~

源码中 Copy-Blend 的主要配置包括：

| 参数 | 作用 |
|---|---|
| copyblend_prob | 启用概率 |
| copyblend_epochs | 生效 Epoch 区间 |
| area_threshold | 目标面积过滤阈值 |
| num_objects | 每张目标图像选取的对象数量 |
| with_expand | 是否扩展裁剪区域 |
| expand_ratios | 扩展比例 |
| copyblend_type | blend 或 copy |

源码中默认的混合系数在 0.45 到 0.55 之间随机选择：

\[
I_{\text{new}}=
\beta I_{\text{target}}
+(1-\beta)I_{\text{source}}
\]

当 copyblend_type 为 blend 时执行混合；否则直接覆盖对应区域。

### 8.4 论文描述与源码实现的区别

论文强调 Copy-Blend 在对象级别增加目标，不像传统 Copy-Paste 那样完全覆盖目标区域。

但官方公开代码中，源对象 Patch 是根据 Bounding Box 裁剪的矩形区域，并没有依赖独立的 Instance Segmentation Mask。

因此：

- Patch 内可能包含目标周围的背景像素。
- 裁剪区域是否足够紧凑，取决于原始标注框质量。
- 它不是逐像素分离前景和背景的实例分割式复制。

这是根据公开源码实现得出的说明，不应把论文中的对象级描述误解成代码一定使用了精确实例 Mask。

### 8.5 MixUp 和 Copy-Blend 并不是连续执行

这是源码中非常值得注意的一个细节。

在 BatchImageCollateFunction 的 apply_mixup 中，MixUp 和 Copy-Blend 使用 if/elif 分支。

因此在一次该函数调用中：

- 如果满足 MixUp 条件，就执行 MixUp。
- 否则才会判断 Copy-Blend 是否执行。

两者不是每个 Batch 都按 MixUp → Copy-Blend 的顺序连续执行。

Mosaic 属于前面的 Dataset Transform，仍然可以与 Batch 级增强策略共同使用。

源码：[engine/data/dataloader.py](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/data/dataloader.py)。

### 8.6 Hungarian Matcher 的匹配成本

DEIMv2 仍然使用 Hungarian Matching。

传统匹配成本可以写成：

\[
C=
\lambda_c C_{\text{class}}
+\lambda_b C_{\text{bbox}}
+\lambda_g C_{\text{GIoU}}
\]

其中：

- \(C_{\text{class}}\)：分类匹配成本。
- \(C_{\text{bbox}}\)：L1 边界框距离。
- \(C_{\text{GIoU}}\)：GIoU 匹配成本。

官方 Matcher 还支持在指定 Epoch 后切换匹配成本。

当 change_matcher 启用并且达到 matcher_change_epoch 后，代码使用：

\[
C=-p_{\text{class}}\cdot IoU^\alpha
\]

其中：

- \(p_{\text{class}}\)：预测类别分数。
- \(IoU\)：预测框与真实框的重叠程度。
- \(\alpha\)：iou_order_alpha，官方配置常见值为 4。

它仍然通过 Hungarian 算法进行分配，只是匹配成本发生变化。

### 8.7 Decoder 不同层的匹配并集

官方 DEIMCriterion 还会收集不同 Decoder Layer 的匹配结果。

代码中通过 get_go_indices 构建跨 Decoder Layer 的匹配并集。

对于 boxes 和 local 损失，默认 use_uni_set=True，会使用这个并集进行监督。

这并不意味着模型在推理时变成了 O2M 检测器，而是训练阶段让多个 Decoder Layer 的有效匹配信息能够参与部分损失计算。

## 九、DEIMv2 的损失函数

DEIMv2 的总损失由五部分组成：

\[
\begin{aligned}
L_{\text{total}}={}&
\lambda_1L_{\text{MAL}}
+\lambda_2L_{\text{FGL}}
+\lambda_3L_{\text{DDF}}\\
&+\lambda_4L_{\text{BBox}}
+\lambda_5L_{\text{GIoU}}
\end{aligned}
\]

论文中的权重：

\[
\begin{aligned}
\lambda_1&=1.0\\
\lambda_2&=0.15\\
\lambda_3&=1.5\\
\lambda_4&=5.0\\
\lambda_5&=2.0
\end{aligned}
\]

### 9.1 MAL：Matchability-Aware Loss

MAL 来自 DEIM。

传统分类损失主要关注预测类别是否正确，而 MAL 还考虑预测框与 Ground Truth 的匹配质量。

对于匹配到的预测框，先计算 IoU：

\[
q_i=IoU(\hat b_i,b_i)
\]

再将这个质量分数用于构造分类监督。

在官方实现中，正样本目标分数与 IoU 的幂次相关：

\[
y_i=q_i^\gamma
\]

负样本的权重则与预测置信度有关。

可以把 MAL 理解为：

- 高质量匹配得到更强的分类目标。
- 低质量匹配不会被当作与高质量匹配完全相同的正样本。
- 高置信度但未匹配的预测会受到相应惩罚。

它将分类与定位质量联系起来。

### 9.2 BBox Loss 与 GIoU Loss

L1 边界框损失：

\[
L_{\text{BBox}}=
\sum_i\|\hat b_i-b_i\|_1
\]

GIoU Loss：

\[
L_{\text{GIoU}}=
1-\operatorname{GIoU}(\hat b_i,b_i)
\]

两者关注点不同：

- L1 关注边界框坐标之间的差异。
- GIoU 关注预测框与真实框之间的几何重叠关系。

### 9.3 FGL：Fine-Grained Localization Loss

FGL 来自 D-FINE 的细粒度定位监督。

前面已经介绍过，Decoder 对一个目标的左、上、右、下四个方向分别预测离散分布。

FGL 将真实边界框转换为对应的分布监督目标，再对预测分布计算 Distribution Focal Loss。

它不是直接对四个连续坐标计算 L1，而是在分布层面学习边界位置。

官方代码中，FGL 的监督权重还会考虑预测框与真实框的 IoU。

### 9.4 DDF：Decoupled Distillation Focal Loss

DDF 也是 D-FINE 相关的局部定位损失。

这里需要纠正一个容易产生的误解：

**DEIMv2 的 DDF 并不意味着又额外加载一个 DINOv3 Teacher 网络。**

官方 Decoder 会将最终 Decoder Layer 的预测分布作为辅助层的 Teacher 输出：

- 当前中间层：Student。
- 最后一层 Decoder：Teacher。

随后通过 KL Divergence 比较中间层和最终层的预测分布。

概念上可以写为：

\[
L_{\text{DDF}}
\sim
T^2
D_{\text{KL}}
\left(
P_{\text{final}}^{(T)}
\parallel
P_{\text{intermediate}}^{(T)}
\right)
\]

其中：

- \(T\)：蒸馏温度，源码默认使用 5。
- \(P_{\text{final}}\)：最后一层 Decoder 的预测分布。
- \(P_{\text{intermediate}}\)：中间层 Decoder 的预测分布。

DDF 的作用是让前面的 Decoder Layer 学习最终层的定位分布，从而改善逐层迭代细化。

### 9.5 为什么 Pico、Femto、Atto 不使用 FGL 和 DDF？

论文给出的实验观察是：在极轻量级模型中加入 FGL、DDF，反而可能导致性能下降。

作者认为，这与模型容量较小、基础预测能力较弱有关。

因此：

| 模型 | MAL | FGL | DDF | BBox | GIoU |
|---|---|---|---|---|---|
| X | ✓ | ✓ | ✓ | ✓ | ✓ |
| L | ✓ | ✓ | ✓ | ✓ | ✓ |
| M | ✓ | ✓ | ✓ | ✓ | ✓ |
| S | ✓ | ✓ | ✓ | ✓ | ✓ |
| Nano | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pico | ✓ |  |  | ✓ | ✓ |
| Femto | ✓ |  |  | ✓ | ✓ |
| Atto | ✓ |  |  | ✓ | ✓ |

这也是为什么不能把 S 模型的 Criterion 配置直接复制到所有轻量级模型。

源码：[engine/deim/deim_criterion.py](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_criterion.py)。

## 十、COCO 实验结果与模型规格对比

### 10.1 八种模型的主要指标

下表使用论文 v5 的结果，并与官方 GitHub 当前 Model Zoo 的数值交叉核对。参数量和 GFLOPs 保留更精确的论文数值。

| 模型 | 输入尺寸 | COCO AP | 参数量 | GFLOPs |
|---|---:|---:|---:|---:|
| Atto | 320 | 23.8 | 0.49M | 0.76 |
| Femto | 416 | 31.0 | 0.96M | 1.67 |
| Pico | 640 | 38.5 | 1.51M | 5.15 |
| Nano | 640 | 43.0 | 3.57M | 6.86 |
| S | 640 | 50.9 | 9.71M | 49.5 |
| M | 640 | 53.0 | 18.11M | 84.1 |
| L | 640 | 56.0 | 32.18M | 144.5 |
| X | 640 | 57.8 | 50.26M | 200.0 |

### 10.2 三个最值得关注的结果

**DEIMv2-S：**

- 参数量 9.71M。
- COCO AP 50.9。
- 作者强调这是首个参数量低于 10M、COCO AP 超过 50 的模型。

**DEIMv2-X：**

- 参数量约 50.3M。
- COCO AP 57.8。
- 对比 DEIM-X 的 56.5 AP 和约 62M 参数，DEIMv2-X 在论文报告的 COCO 指标上更好，参数量更低。

**DEIMv2-Pico：**

- 参数量约 1.5M。
- COCO AP 38.5。
- 与 YOLOv10-Nano 的 38.5 AP 相当，但参数量更低。

这些结果体现了 DEIMv2 的一个特点：它不是只提供一个高性能模型，而是从极轻量到大模型覆盖了多个计算预算。

### 10.3 一个对小目标检测非常重要的观察

论文还比较了 DEIMv2 和原始 DEIM 在不同目标尺寸上的 AP。

以 S 模型为例：

| 指标 | DEIM-S | DEIMv2-S | 提升 |
|---|---:|---:|---:|
| AP | 49.0 | 50.9 | +1.9 |
| AP_S | 30.4 | 31.4 | +1.0 |
| AP_M | 52.6 | 55.3 | +2.7 |
| AP_L | 65.7 | 70.3 | +4.6 |

以 X 模型为例：

| 指标 | DEIM-X | DEIMv2-X | 提升 |
|---|---:|---:|---:|
| AP | 56.5 | 57.8 | +1.3 |
| AP_S | 38.8 | 39.2 | +0.4 |
| AP_M | 61.4 | 62.8 | +1.4 |
| AP_L | 74.2 | 75.9 | +1.7 |

作者在论文中指出：

**DINOv3 路线的性能提升主要来自中、大目标，小目标上的收益相对有限。**

这说明 DINOv3 的强语义特征并不意味着小目标检测一定会大幅提升。

对于密集植物、小目标和严重遮挡场景，不能只看 COCO 总 AP，还需要单独检查：

- AP_S 或自定义的小目标 AP。
- Recall。
- 密集目标漏检率。
- 相邻目标是否被合并。
- 目标边缘和细节是否保留。

### 10.4 FLOPs 数据的版本变化

官方 GitHub 在 2026-10-22 更新说明中指出，之前使用的 FLOPs 统计逻辑遗漏了 Attention FLOPs，之后修正了计算方式。

因此，早期论文版本或缓存页面中可能出现：

| 模型 | 早期统计 | 修正后的 GFLOPs |
|---|---:|---:|
| S | 25.62 | 49.5 |
| M | 52.20 | 84.1 |
| L | 96.32 | 144.5 |
| X | 151.6 | 200.0 |

本文采用论文 v5 和当前官方仓库修正后的数值。

引用 DEIMv2 的 FLOPs 时，应明确使用哪个版本的数据，不能把早期统计值和当前参数量混在一起。

### 10.5 为什么本文不把 Latency 作为主要排名指标？

论文 v5 和官方 README 都提供了 Latency，但两者在 Atto、Femto 的数值上存在差异。

另外，论文也指出 ViT 路线的推理延迟尚未经过充分优化。

如果缺少统一的硬件、推理框架、精度和测试设置，Latency 不能直接当作跨设备的公平速度比较。

因此本文不使用 Latency 对模型进行排名。实际项目应在目标 GPU、CPU 或边缘设备上重新测试。

## 十一、训练配置与自定义数据集

### 11.1 环境准备

官方仓库给出的测试环境参考：

- Python 3.11。
- PyTorch 2.4.1 或 2.5.1。
- 推荐 PyTorch 2.0 及以上。

安装：

~~~bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git
cd DEIMv2

conda create -n deimv2 python=3.11 -y
conda activate deimv2

pip install -r requirements.txt
~~~

### 11.2 不同 Backbone 需要哪些预训练权重？

| 模型 | Backbone 权重 |
|---|---|
| S | vitt_distill.pt |
| M | vittplus_distill.pt |
| L | DINOv3 ViT-Small |
| X | DINOv3 ViT-Small+ |
| Nano / Pico / Femto / Atto | HGNetv2 权重，官方配置支持自动下载 |

L、X 的官方配置使用：

~~~text
ckpts/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
~~~

S、M 使用作者从 DINOv3 ViT-Small 蒸馏得到的 Tiny、Tiny+ 权重：

~~~text
ckpts/
├── vitt_distill.pt
└── vittplus_distill.pt
~~~

S、M 的蒸馏过程并没有在 DEIMv2 论文中完整展开，因此不要自行推断蒸馏数据、训练轮数或全部损失细节。

### 11.3 自定义数据集格式

官方仓库支持 COCO 格式目标检测数据集。

~~~text
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
~~~

自定义类别时，修改数据集配置：

~~~yaml
remap_mscoco_category: False
~~~

并设置：

~~~yaml
num_classes: 你的类别数量
~~~

同时修改训练和验证数据路径：

~~~yaml
train_dataloader:
  dataset:
    img_folder: /data/yourdataset/train
    ann_file: /data/yourdataset/train/train.json

val_dataloader:
  dataset:
    img_folder: /data/yourdataset/val
    ann_file: /data/yourdataset/val/ann.json
~~~

如果原始标注不是 COCO 格式，需要先转换成 COCO Detection 标注。

### 11.4 训练命令

DINOv3 路线：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--use-amp \
--seed=0
~~~

HGNetv2 路线：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_hgnetv2_pico_coco.yml \
--use-amp \
--seed=0
~~~

测试已有权重：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--test-only \
-r model.pth
~~~

微调已有权重：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--use-amp \
--seed=0 \
-t model.pth
~~~

训练命令中的配置文件需要与模型规格对应，不能只替换 checkpoint 文件名。

### 11.5 训练 Epoch 与 EMA

论文中的 Epoch 数量与官方 YAML 中的 epoches 不一定相等。

例如 S 模型：

- 论文训练 Epoch：120。
- 官方配置 epoches：132。
- 额外的 12 个 Epoch 与 EMA 训练安排有关。

官方 README 给出的简化示例中，20 个实际训练 Epoch 可以设置为：

~~~yaml
epoches: 32
flat_epoch: 14
no_aug_epoch: 12
~~~

修改训练 Epoch 时，还需要同步调整：

- flat_epoch。
- no_aug_epoch。
- Mosaic 生效区间。
- MixUp 生效区间。
- Copy-Blend 生效区间。
- matcher_change_epoch。
- EMA 相关设置。

不能只修改 epoches 一个参数。

### 11.6 论文中的训练设置

| 模型 | 输入尺寸 | 训练 Epoch | Local Loss | Mosaic 概率 | MixUp 概率 | Copy-Blend 概率 |
|---|---:|---:|---|---:|---:|---:|
| X | 640 | 50 | ✓ | 0.5 | 0.5 | 0.5 |
| L | 640 | 60 | ✓ | 0.5 | 0.5 | 0.5 |
| M | 640 | 90 | ✓ | 0.5 | 0.5 | 0.5 |
| S | 640 | 120 | ✓ | 0.5 | 0.5 | 0.5 |
| Nano | 640 | 148 | ✓ | 0.5 | 0.5 | 0.4 |
| Pico | 640 | 468 |  | 0.5 | 0.0 | 0.0 |
| Femto | 416 | 468 |  | 0.5 | 0.0 | 0.0 |
| Atto | 320 | 468 |  | 0.3 | 0.0 | 0.0 |

Local Loss 指 FGL 和 DDF。

这里可以看到，超轻量模型并没有机械地使用所有增强和损失。

## 十二、Hugging Face 模型权重与加载方式

### 12.1 作者发布的官方模型

作者在 Intellindust 组织中发布了 DEIMv2 的模型权重。

| 模型 | Hugging Face |
|---|---|
| Atto | [DEIMv2_HGNetv2_ATTO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO) |
| Femto | [DEIMv2_HGNetv2_FEMTO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO) |
| Pico | [DEIMv2_HGNetv2_PICO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO) |
| Nano | [DEIMv2_HGNetv2_N_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO) |
| S | [DEIMv2_DINOv3_S_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO) |
| M | [DEIMv2_DINOv3_M_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO) |
| L | [DEIMv2_DINOv3_L_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO) |
| X | [DEIMv2_DINOv3_X_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO) |

这些官方仓库使用 Safetensors 和 PyTorchModelHubMixin。

### 12.2 官方仓库的加载方式

官方 README 和 hf_models.ipynb 提供了模型封装示例。

首先需要 Clone 官方仓库，保证本地可以导入 engine 模块：

~~~python
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        else:
            self.backbone = HGNetv2(**config["HGNetv2"])

        if "LiteEncoder" in config:
            self.encoder = LiteEncoder(**config["LiteEncoder"])
        else:
            self.encoder = HybridEncoder(**config["HybridEncoder"])

        if "DEIMTransformer" in config:
            self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        else:
            self.decoder = DFINETransformer(**config["DFINETransformer"])

        self.postprocessor = PostProcessor(**config["PostProcessor"])

    def forward(self, x, orig_target_sizes):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)
        return x
~~~

加载作者官方 S 模型：

~~~python
model = DEIMv2.from_pretrained(
    "Intellindust/DEIMv2_DINOv3_S_COCO"
)
~~~

这段封装代码来自官方仓库的 Hugging Face 使用示例。实际加载时仍需要与模型对应的配置对象，建议直接参考仓库中的 hf_models.ipynb，使用对应规格的配置和模型类。

### 12.3 Transformers 文档与作者原始权重不是一回事

Hugging Face Transformers 文档提供了：

- Deimv2Config。
- Deimv2ForObjectDetection。

但需要注意：

- 作者 Intellindust 发布的原始模型使用 PyTorchModelHubMixin。
- Transformers 文档中的示例还涉及第三方转换后的模型，例如 harshaljanjani 发布的 Transformers 版本。
- 第三方转换模型与作者原始权重不应默认视为完全相同的文件格式和加载流程。

如果目标是复现论文或使用作者发布的 checkpoint，优先参考官方 GitHub 仓库和对应 Hub 模型配置。

## 十三、ONNX 与 TensorRT 部署

官方仓库提供 ONNX 导出、ONNX Runtime、TensorRT 和 PyTorch 推理脚本。

### 13.1 ONNX 导出

安装依赖：

~~~bash
pip install onnx onnxsim
~~~

导出：

~~~bash
python tools/deployment/export_onnx.py \
--check \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
-r model.pth
~~~

导出脚本默认使用 ONNX Opset 17。

### 13.2 ONNX 输入和输出

官方导出脚本中的输入为：

| 输入 | 维度 | 含义 |
|---|---|---|
| images | N × 3 × H × W | 输入图像 |
| orig_target_sizes | N × 2 | 每张图像的原始高、宽 |

输出为：

| 输出 | 含义 |
|---|---|
| labels | 类别索引 |
| boxes | 边界框 |
| scores | 置信度 |

导出脚本将模型和 PostProcessor 封装在一起。

当前脚本的动态轴主要设置在 Batch 维度，不能默认认为输入 H、W 可以任意动态变化。若需要改变输入尺寸，应同步修改 eval_spatial_size、预处理和导出设置。

### 13.3 后处理是否需要 NMS？

DEIMv2 是 DETR 风格的集合预测检测器，核心后处理不依赖 YOLO 式标准 NMS。

Focal Loss 模式下，官方 PostProcessor 主要执行：

1. 对分类 Logits 使用 Sigmoid。
2. 将 Query 和类别维度展平。
3. 选择 Top-K 分数。
4. 根据索引恢复类别和 Query 索引。
5. 取出对应预测框。
6. 根据 orig_target_sizes 将归一化坐标恢复到原图像素坐标。

因此部署时需要重点检查：

- Top-K 的数量。
- 类别索引的计算。
- Query 索引的恢复。
- 边界框从 cxcywh 到 xyxy 的转换。
- 原图尺寸的缩放。

不要直接把 YOLO 的 NMS 后处理代码套用到 DEIMv2。

源码：[engine/deim/postprocessor.py](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/postprocessor.py)。

### 13.4 TensorRT 部署

转换：

~~~bash
trtexec \
--onnx="model.onnx" \
--saveEngine="model.engine" \
--fp16
~~~

官方仓库特别提示：

- 推荐 TensorRT 10.6 或更高版本进行 FP16 推理。
- TensorRT 10.4 的 FP16 推理可能产生错误检测输出。
- 如果使用旧版本，需要考虑 FP32，并对整个推理流程进行数值和检测精度验证。

官方还提供：

- tools/inference/torch_inf.py
- tools/inference/onnx_inf.py
- tools/inference/trt_inf.py
- tools/benchmark/get_info.py
- tools/benchmark/trt_benchmark.py

这些工具分别用于 PyTorch、ONNX、TensorRT 推理，以及模型参数量、FLOPs 和实际延迟测试。

## 十四、DEIMv2 的优势、局限与适用场景

### 14.1 优势

1. **DINOv3 特征迁移：** 通过 STA 将基础模型的语义表征转成检测所需的多尺度特征。
2. **模型规格丰富：** 从 Atto 到 X，覆盖多个计算预算。
3. **轻量化路线明确：** 极低算力场景不必强行使用 ViT，而可以选择 HGNetv2。
4. **训练监督更充分：** Dense O2O、MAL 和 Copy-Blend 从不同角度增强训练。
5. **Decoder 更高效：** SwiGLU、RMSNorm、共享 Query Position Embedding 等设计减少冗余计算。
6. **部署路径完整：** 官方提供 ONNX 和 TensorRT 的导出与推理工具。

### 14.2 局限与注意事项

1. DINOv3 路线依赖相应预训练或蒸馏 Backbone 权重，不能随意替换。
2. DINOv3 的全局语义优势不代表小目标 AP 一定大幅提升。
3. STA 增加了空间先验分支和融合计算，实际推理速度需要在目标设备上测量。
4. Nano 与 Pico/Femto/Atto 的 Encoder 路线不同，不能只按参数量判断模型结构。
5. Copy-Blend 依赖目标框质量，矩形 Patch 可能带入框内背景。
6. 论文、官方 YAML 和 Hub 配置存在个别参数差异，复现时必须锁定版本和配置。
7. COCO 上的结果不代表自定义数据集可以直接获得相同 AP。

### 14.3 如何选择模型？

| 使用条件 | 可以优先评估 |
|---|---|
| 极低算力、模型体积极小 | Atto / Femto |
| 轻量边缘设备 | Pico / Nano |
| 中等算力 GPU，希望兼顾精度 | S / M |
| 高精度 GPU 检测 | L / X |

最终选择需要在目标数据集和目标硬件上评估：

- AP 和 Recall。
- 小目标 AP。
- 密集场景漏检率。
- 参数量和显存。
- FP16/FP32 实际推理延迟。
- ONNX/TensorRT 算子兼容性。

## 十五、总结

DEIMv2 的核心价值是把视觉基础模型与实时 DETR 检测器结合，并针对不同硬件预算设计完整模型系列。

它的关键设计可以归纳为：

- DINOv3 提供高质量语义特征。
- STA 将 ViT 单尺度特征适配为多尺度检测特征。
- HGNetv2 裁剪路线覆盖极轻量级设备。
- HybridEncoder 和 LiteEncoder 适配不同 Backbone 输出。
- DEIM Transformer Decoder 使用 Deformable Attention、SwiGLU、RMSNorm 和共享位置编码。
- D-FINE 风格分布回归改善边界框定位。
- Dense O2O、MAL 和 Copy-Blend 改善训练监督。

需要特别记住两个结论：

1. **DEIMv2 并不是单纯依赖 DINOv3 提升检测精度。** 它的效果来自 Backbone、特征适配、Decoder、匹配策略和数据增强的组合。
2. **DINOv3 带来的收益并不均匀。** 论文显示中、大目标提升更明显，小目标仍然是需要单独研究的问题。

对于实际项目，应该先确定硬件预算和目标尺寸分布，再选择模型规格。不要只根据 COCO 总 AP 或参数量决定模型。

## 参考资料

### 论文

1. [Real-Time Object Detection Meets DINOv3（DEIMv2，最新 v5）](https://arxiv.org/pdf/2509.20787v5)
2. [DEIM: DETR with Improved Matching for Fast Convergence（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html)
3. [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)
4. [DINOv3](https://arxiv.org/abs/2508.10104)

### 官方 GitHub 与源码

5. [DEIMv2 官方仓库](https://github.com/Intellindust-AI-Lab/DEIMv2)
6. [官方 README / Model Zoo](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/README.md)
7. [DINOv3 STA Adapter](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/dinov3_adapter.py)
8. [ViT-Tiny Backbone](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/vit_tiny.py)
9. [HybridEncoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/hybrid_encoder.py)
10. [LiteEncoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/lite_encoder.py)
11. [DEIM Decoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py)
12. [DEIM Criterion](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_criterion.py)
13. [Hungarian Matcher](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/matcher.py)
14. [Copy-Blend 实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/data/dataloader.py)
15. [PostProcessor](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/postprocessor.py)
16. [ONNX 导出脚本](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/tools/deployment/export_onnx.py)

### 模型配置与 Hugging Face

17. [DINOv3-S 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_s_coco.yml)
18. [DINOv3-M 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_m_coco.yml)
19. [DINOv3-L 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_l_coco.yml)
20. [DINOv3-X 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml)
21. [Pico 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_hgnetv2_pico_coco.yml)
22. [官方 Hugging Face 模型组织](https://huggingface.co/Intellindust)
23. [官方 Hugging Face 使用 Notebook](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/hf_models.ipynb)
24. [Hugging Face Transformers DEIMv2 文档](https://huggingface.co/docs/transformers/model_doc/deimv2)
