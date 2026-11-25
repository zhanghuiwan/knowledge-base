---
title: DEIMv2 深度解析：DINOv3 特征如何适配实时 DETR
description: 从论文 v5、官方源码、模型配置和 Hugging Face 权重出发，拆解 DEIMv2 的双 Backbone 路线、STA、Hybrid/Lite Encoder、Transformer Decoder、Dense O2O、Copy-Blend、损失函数、实验结果与部署。
---

# DEIMv2 深度解析：DINOv3 特征如何适配实时 DETR

> 本文从论文、官方实现和模型配置三个层面分析 DEIMv2。重点回答：它由哪些网络模块组成？DINOv3 的单尺度特征如何变成检测所需的多尺度特征？DEIM 的 Dense O2O 和 MAL 如何训练？Decoder 如何产生边界框？论文指标与当前代码配置有哪些差异？

**资料版本**

- 论文：*Real-Time Object Detection Meets DINOv3*，使用 arXiv v5，修订日期为 2026-10-27。
- 官方代码：Intellindust-AI-Lab/DEIMv2，本文核对的 main 提交为 [f005f50](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/f005f50ce565e63f6c298feccb51901a98c3d035)，日期为 2026-10-23。
- 模型配置：以该提交中的 YAML 为主；权重结构另外核对作者的 Hugging Face 配置。
- 指标：优先使用论文 v5 和官方仓库 2026-10-22 修正后的 FLOPs 数据。

论文、源码和模型卡不是同一个概念：论文描述研究方法与实验结果，YAML 描述可运行配置，模型卡描述已发布权重的结构。遇到不一致时，本文分别说明，不把它们拼成一个虚构的“统一版本”。

## 一、先理解 DEIMv2 是什么

DEIMv2 的论文标题是 **Real-Time Object Detection Meets DINOv3**。它建立在作者此前的 DEIM 训练框架上，将 DINOv3 的视觉特征引入实时 DETR 检测器，同时保留面向轻量部署的 CNN 路线。

它不是简单地把 DINOv3 接到一个 DETR Head 上。完整系统包含四类设计：

1. **两条 Backbone 路线**：S、M、L、X 使用 DINOv3 预训练或蒸馏得到的 ViT；Nano、Pico、Femto、Atto 使用不同规模的 HGNetv2。
2. **不同的多尺度特征适配方式**：ViT 路线使用 Spatial Tuning Adapter（STA）；HGNetv2 路线使用 HybridEncoder 或 LiteEncoder。
3. **高效的 Transformer Decoder**：使用多尺度可变形交叉注意力、SwiGLU FFN、RMSNorm、共享 Query Position Embedding，并保留 D-FINE 风格的分布式边界框细化。
4. **改进训练监督**：继承 DEIM 的 Dense One-to-One（Dense O2O）和 Matchability-Aware Loss（MAL），并增加对象级 Copy-Blend 数据增强。

可以先用一句话概括：

**DEIMv2 = RT-DETR 风格的检测框架 +（DINOv3/STA 或轻量 HGNetv2）+ DEIM 训练策略 + D-FINE 风格的定位细化。**

这里的“+”表示技术组合，不代表这些方法由同一篇论文首次提出。

### 1.1 论文、代码与模型入口

| 资料 | 用途 |
|---|---|
| [DEIMv2 论文 v5](https://arxiv.org/abs/2509.20787v5) | 方法、实验结果和模型规格 |
| [论文 PDF v5](https://arxiv.org/pdf/2509.20787v5) | 阅读原始公式、图和表格 |
| [DEIMv2 官方 GitHub](https://github.com/Intellindust-AI-Lab/DEIMv2) | 训练、配置、推理和部署 |
| [官方代码核对版本](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/f005f50ce565e63f6c298feccb51901a98c3d035) | 本文分析所依据的代码提交 |
| [DEIMv2 项目页面](https://intellindust-ai-lab.github.io/projects/DEIMv2/) | 项目介绍 |
| [Intellindust Hugging Face 组织](https://huggingface.co/Intellindust) | 作者发布的检测模型权重 |
| [DINOv3 论文](https://arxiv.org/abs/2508.10104) | 理解预训练 Backbone 的来源 |
| [DINOv3 官方代码](https://github.com/facebookresearch/dinov3) | 理解 DINOv3 的模型实现 |

### 1.2 DEIM、RT-DETR、D-FINE、DINOv3 分别负责什么

这些名称处于不同的技术层次，不应混为一谈。

| 方法 | 主要解决的问题 | 在 DEIMv2 中的关系 |
|---|---|---|
| RT-DETR | 实时端到端检测器的整体组织方式，包括 Backbone、Hybrid Encoder、Decoder | DEIMv2 沿用类似的整体组织方式 |
| DEIM | 通过 Dense O2O 和 MAL 改善 DETR 的训练监督与收敛 | DEIMv2 继承并扩展这套训练策略 |
| D-FINE | 将边界框回归改造成细粒度分布细化，并引入定位自蒸馏 | DEIMv2 的 Decoder 和局部定位损失采用相关设计 |
| DINOv3 | 自监督视觉预训练模型，提供通用视觉特征 | 为 S/M/L/X 提供预训练或蒸馏 Backbone |
| HGNetv2 | 面向实时检测的高效 CNN Backbone | 为 Nano/Pico/Femto/Atto 提供轻量特征 |

**特别注意：DEIMv2 中的 DINOv3 与目标检测论文 DINO 不是同一个模型。** 这里使用的是自监督视觉表征模型 DINOv3，而不是 *DETR with Improved DeNoising Anchor Boxes*。

### 1.3 DEIM 与 DEIMv2 的区别

DEIM（CVPR 2025）的主要贡献在训练机制：

- Dense O2O：通过数据增强增加每张训练图像中的目标数量，从而增加一对一匹配监督的数量。
- MAL：针对 Dense O2O 带来的低质量匹配，设计能够感知匹配质量的分类损失。

DEIMv2 在此基础上进一步改进模型结构与训练数据：

- 将 DINOv3 或其蒸馏得到的 ViT 特征用于较大模型。
- 使用 STA 将 ViT 特征适配为检测所需的多尺度特征。
- 为超轻量级模型设计裁剪后的 HGNetv2 分支。
- 使用 SwiGLU、RMSNorm 和共享 Query Position Embedding 简化 Decoder。
- 在对象级别增加 Copy-Blend。

因此，DEIMv2 既不是一个单独的 Backbone，也不是单纯的训练技巧，而是一套完整的检测模型家族。

## 二、整体网络架构与八种模型规格

DEIMv2 的整体前向过程可以写成：

$$
\text{Image}
\rightarrow
\text{Backbone}
\rightarrow
\text{Encoder}
\rightarrow
\text{Decoder}
\rightarrow
\text{PostProcessor}
$$

### 2.1 两条 Backbone 路线

~~~mermaid
flowchart TD
    A["输入图像"] --> B{"模型规格"}
    B -->|"S / M / L / X"| C["DINOv3 或蒸馏 ViT-Tiny"]
    C --> D["STA：语义特征与空间细节融合"]
    D --> E["1/8、1/16、1/32 三尺度特征"]
    E --> F["HybridEncoder"]
    B -->|"Nano"| G["HGNetv2-B0"]
    G --> H["1/16、1/32 两尺度特征"]
    H --> I["HybridEncoder"]
    B -->|"Pico / Femto / Atto"| J["裁剪 HGNetv2"]
    J --> K["1/16 特征"]
    K --> L["LiteEncoder"]
    L --> M["1/16、1/32 两尺度特征"]
    F --> N["DEIM Transformer Decoder"]
    I --> N
    M --> N
    N --> O["分类分数与边界框"]
    O --> P["Top-K 后处理"]
    P --> Q["检测结果"]
~~~

两条路线的差异：

- **ViT 路线**：DINOv3 的语义特征较强，但原生 ViT 输出是单尺度 Patch 特征，需要 STA 补充局部细节并构建多尺度表示。
- **HGNetv2 路线**：直接使用 CNN 的层级特征。Nano 已有两个 Backbone 输出尺度，Pico/Femto/Atto 则只保留一个主要尺度，再由 LiteEncoder 生成第二个尺度。

### 2.2 八种官方模型规格

下表中的输入尺寸、参数量、GFLOPs 和 AP 以论文 v5 的模型规格表及官方仓库修正后的 Model Zoo 为准。参数量和计算量经过四舍五入。

| 规格 | Backbone | Encoder | 特征尺度数 | Decoder 层数 | Query 数 | 输入尺寸 | 参数量 | GFLOPs | COCO AP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Atto | HGNetv2-Atto | LiteEncoder | 2 | 3 | 100 | 320 | 0.49M | 0.76 | 23.8 |
| Femto | HGNetv2-Femto | LiteEncoder | 2 | 3 | 150 | 416 | 0.96M | 1.67 | 31.0 |
| Pico | HGNetv2-Pico | LiteEncoder | 2 | 3 | 200 | 640 | 1.51M | 5.15 | 38.5 |
| Nano（N） | HGNetv2-B0 | HybridEncoder | 2 | 3 | 300 | 640 | 3.57M | 6.86 | 43.0 |
| S | ViT-Tiny | HybridEncoder | 3 | 4 | 300 | 640 | 9.71M | 49.5 | 50.9 |
| M | ViT-Tiny+ | HybridEncoder | 3 | 4 | 300 | 640 | 18.11M | 84.1 | 53.0 |
| L | DINOv3 ViT-Small | HybridEncoder | 3 | 4 | 300 | 640 | 32.18M | 144.5 | 56.0 |
| X | DINOv3 ViT-Small+ | HybridEncoder | 3 | 6 | 300 | 640 | 50.26M | 200.0 | 57.8 |

这里的 Query 数是 Decoder 使用的 Object Query 数量，不等于最终输出的检测框数量。后处理还会根据分类分数进行 Top-K 筛选。

### 2.3 为什么模型家族采用两种 Backbone？

DINOv3 路线主要服务于对精度要求较高的场景：

- L/X 使用官方 DINOv3 ViT-Small / ViT-Small+ 预训练 Backbone。
- S/M 使用从 DINOv3 ViT-Small 蒸馏得到的 ViT-Tiny / ViT-Tiny+。
- 四种模型都使用 STA 和三尺度特征。

HGNetv2 路线主要服务于极端资源受限场景：

- Nano 使用 HGNetv2-B0。
- Pico 在 B0 的基础上移除最后一个 Stage，只保留到 1/16 尺度的输出。
- Femto 进一步减少 Pico 最后一个 Stage 的 Block 数量。
- Atto 再缩减最后一个 Block 的通道数。

这意味着不能简单认为所有 DEIMv2 都使用 DINOv3。Atto、Femto、Pico 和 Nano 并不走 DINOv3 Backbone 路线。

## 三、DINOv3 Backbone 与 Spatial Tuning Adapter（STA）

这是 DEIMv2 最有代表性的结构改进，也是理解 ViT 路线的关键。

### 3.1 为什么不能直接把 DINOv3 输出送入检测头？

DINOv3 是基于 Vision Transformer 的视觉表征模型。ViT 将输入图像切分为 Patch，再将 Patch 映射成 Token 序列，经过 Transformer Block 提取特征。

假设输入图片为：

$$
X\in\mathbb{R}^{B\times3\times640\times640}
$$

Patch Size 为 16，则 Patch 网格大小为：

$$
H_p=W_p=\frac{640}{16}=40
$$

Patch Token 数量为：

$$
N=40\times40=1600
$$

以隐藏维度 $D$ 为例，去除 CLS Token 后的 Patch 特征可以还原为：

$$
F_{\text{ViT}}\in\mathbb{R}^{B\times D\times40\times40}
$$

这对应 1/16 尺度的特征。

检测目标的尺寸差异很大。小目标需要较高分辨率的特征，大目标则需要更大的感受野和更强的语义信息。检测网络通常需要多个尺度的特征图，例如：

- 1/8：保留相对丰富的空间细节。
- 1/16：兼顾空间细节与语义。
- 1/32：具有更大的感受野和更强的高层语义。

如果只使用 DINOv3 最后一层的 1/16 特征，就需要额外的结构将其适配为检测所需的多尺度特征。

### 3.2 STA 的核心思想

STA 全称 Spatial Tuning Adapter。它并行使用两条路径：

1. **语义路径**：从 DINOv3 不同深度的 Transformer Block 提取 Patch 特征。
2. **空间细节路径**：将原始 RGB 图像送入轻量 CNN，逐步下采样，提取具有较小感受野的局部细节。

最后将两条路径在相同空间尺度上融合。

~~~mermaid
flowchart TD
    A["输入图像 640×640"] --> B["DINOv3 ViT"]
    A --> C["Spatial Prior Module"]
    B --> D["选取多个中间 Block 特征"]
    D --> E["Patch Token 还原为 40×40 特征图"]
    E --> F["双线性插值"]
    F --> G1["80×80 语义特征"]
    F --> G2["40×40 语义特征"]
    F --> G3["20×20 语义特征"]
    C --> H1["1/8 CNN 细节特征"]
    C --> H2["1/16 CNN 细节特征"]
    C --> H3["1/32 CNN 细节特征"]
    G1 --> I1["Concat + 1×1 Conv + Norm"]
    H1 --> I1
    G2 --> I2["Concat + 1×1 Conv + Norm"]
    H2 --> I2
    G3 --> I3["Concat + 1×1 Conv + Norm"]
    H3 --> I3
    I1 --> J["多尺度检测特征"]
    I2 --> J
    I3 --> J
~~~

注意：论文将 ViT 中间特征的尺度变换描述为 parameter-free resizing。这里的“无参数”指双线性插值本身不需要学习参数，**并不意味着整个 STA 没有可学习参数**。官方实现中的 Spatial Prior Module、1×1 Conv 和归一化层都是有参数的。

### 3.3 语义路径：提取不同深度的 ViT 特征

DEIMv2 不只使用最后一个 Transformer Block 的输出，而是选择多个中间层。

当前官方配置中：

| 规格 | ViT 类型 | interaction_indexes | Patch 特征维度 |
|---|---|---|---:|
| S | ViT-Tiny | [3, 7, 11] | 192 |
| M | ViT-Tiny+ | [3, 7, 11] | 256 |
| L | DINOv3 ViT-Small | [5, 8, 11] | 384 |
| X | DINOv3 ViT-Small+ | [5, 8, 11] | 384 |

这些索引是 Python 配置中的层索引，不应在没有核对具体实现时直接将其称为自然语言中的第几层。

每个被选中的 ViT 特征先从 Token 序列还原成二维特征图，再通过双线性插值形成不同尺度：

$$
F_{\text{ViT}}^{(i)}
\in\mathbb{R}^{B\times D\times40\times40}
$$

$$
\begin{aligned}
F_{\text{sem}}^{1/8} &= \operatorname{Resize}(F_{\text{ViT}}^{(i_1)},80,80)\\
F_{\text{sem}}^{1/16} &= \operatorname{Resize}(F_{\text{ViT}}^{(i_2)},40,40)\\
F_{\text{sem}}^{1/32} &= \operatorname{Resize}(F_{\text{ViT}}^{(i_3)},20,20)
\end{aligned}
$$

这里不是将同一层特征复制三份，而是分别使用不同深度的 Transformer 特征，再调整空间分辨率。

### 3.4 空间细节路径：Spatial Prior Module

官方代码中的 SpatialPriorModulev2 是一个轻量 CNN，直接处理原始输入图像。

以默认 inplanes=16 为例：

| 操作 | 空间尺寸（输入 640） | 输出通道 |
|---|---:|---:|
| Conv 3×3，Stride 2 | 320×320 | 16 |
| MaxPool 3×3，Stride 2 | 160×160 | 16 |
| Conv 3×3，Stride 2 | 80×80 | 32 |
| Conv 3×3，Stride 2 | 40×40 | 64 |
| Conv 3×3，Stride 2 | 20×20 | 64 |

最终返回：

$$
F_{\text{detail}}=
\{F_d^{1/8},F_d^{1/16},F_d^{1/32}\}
$$

对应形状为：

$$
\begin{aligned}
F_d^{1/8}&\in\mathbb{R}^{B\times32\times80\times80}\\
F_d^{1/16}&\in\mathbb{R}^{B\times64\times40\times40}\\
F_d^{1/32}&\in\mathbb{R}^{B\times64\times20\times20}
\end{aligned}
$$

L 和 X 使用更大的 conv_inplane，因此细节分支的通道数也会相应变化。

### 3.5 Bi-Fusion：语义与细节如何融合？

将相同尺度的语义特征与细节特征在通道维度拼接：

$$
F_{\text{cat}}^s=
\operatorname{Concat}(F_{\text{sem}}^s,F_{\text{detail}}^s)
$$

再使用 1×1 卷积投影到检测网络需要的通道数：

$$
F_{\text{out}}^s=
\operatorname{Norm}(\operatorname{Conv}_{1\times1}(F_{\text{cat}}^s))
$$

例如 S 模型的隐藏维度为 192，默认 conv_inplane=16：

- 1/8 尺度拼接通道数：$192+32=224$。
- 1/16 尺度拼接通道数：$192+64=256$。
- 1/32 尺度拼接通道数：$192+64=256$。

每个尺度再经过独立的 1×1 Conv 和 SyncBatchNorm，输出 192 通道。

**STA 的本质不是简单地把 ViT 特征上采样，而是将高层语义特征与从原始图像提取的局部细节结合。** 这也是它区别于仅使用 Feature Pyramid Upsampling 的关键。

论文 Fig.2 中 Bi-Fusion 小图里的 Bilinear 对应语义特征的双线性插值；官方代码在融合前通过 `F.interpolate(..., mode="bilinear")` 调整 ViT 特征的空间尺寸。它不是一个额外的可学习 Bilinear Layer。之后才将插值后的语义特征与 CNN 细节特征拼接，并通过 1×1 Conv 投影。

但上采样不会凭空恢复 ViT Patch Embedding 之前已经丢失的细节。STA 的 CNN 分支能够补充原图局部信息，却不能保证极小目标一定被保留；后面会结合实验结果讨论这一点。

### 3.6 S/M 与 L/X 的预训练差异

| 规格 | Backbone 初始化来源 | 训练方式 |
|---|---|---|
| S | 从 DINOv3 ViT-Small 蒸馏得到的 ViT-Tiny | 在检测训练中微调 |
| M | 从 DINOv3 ViT-Small 蒸馏得到的 ViT-Tiny+ | 在检测训练中微调 |
| L | 官方 DINOv3 ViT-Small 预训练权重 | 在检测训练中微调 |
| X | 官方 DINOv3 ViT-Small+ 预训练权重 | 在检测训练中微调 |

论文说明了 Tiny/Tiny+ 的蒸馏来源，但没有在 DEIMv2 论文中完整公开这两套蒸馏模型的训练数据、损失配置和全部蒸馏细节。因此不能仅凭 DEIMv2 检测论文推导出完整的 Tiny 蒸馏训练方案。

## 四、HGNetv2 轻量路线与 Encoder

不是所有模型都使用 STA。Nano、Pico、Femto、Atto 采用 CNN Backbone，之后根据特征层数选择不同 Encoder。

### 4.1 Nano：HGNetv2-B0 + HybridEncoder

Nano 使用 HGNetv2-B0，取两个尺度的 Backbone 输出：

- 1/16 特征：40×40。
- 1/32 特征：20×20。

随后通过 HybridEncoder 进行特征交互与融合。

### 4.2 Pico、Femto、Atto：裁剪 HGNetv2 + LiteEncoder

这三个模型只从 Backbone 获取一个主要的 1/16 特征。LiteEncoder 先统一通道，再下采样生成 1/32 特征。

~~~mermaid
flowchart TD
    A["HGNetv2 单尺度输出：1/16"] --> B["1×1 Conv 通道投影"]
    B --> C["AvgPool + Conv 下采样"]
    C --> D["1/32 特征"]
    D --> E["GAP Fusion"]
    B --> F["1/16 特征"]
    E --> G["FPN：上采样并融合"]
    F --> G
    G --> H["PAN：下采样并融合"]
    H --> I["输出 1/16、1/32 特征"]
~~~

LiteEncoder 的关键操作包括：

1. input_proj：通过 1×1 Conv 将输入通道统一到 hidden_dim。
2. down_sample1：通过 AvgPool、1×1 Conv 和归一化生成较低分辨率特征。
3. GAP_Fusion：对特征做 Global Average Pooling，将全局信息加回原特征，再通过卷积处理。
4. FPN 路径：将 1/32 特征上采样，与 1/16 特征相加，再经过 RepNCSPELAN 模块。
5. PAN 路径：将融合后的高分辨率特征下采样，与低分辨率特征融合。

LiteEncoder 不包含与三尺度 ViT 路线相同的 Transformer Encoder 全局交互结构，目的是降低极轻量模型的计算开销。

### 4.3 HybridEncoder：先做尺度内交互，再做跨尺度融合

HybridEncoder 主要由两部分组成：

- Intra-scale Feature Interaction：在选定尺度内通过 Transformer Encoder 进行全局信息交互。
- Cross-scale Feature Fusion：通过 FPN/PAN 风格的 CNN 路径融合不同尺度。

对于 ViT 路线，输入为：

$$
\{F_3,F_4,F_5\}
$$

其形状以 S 模型、640 输入为例：

$$
\begin{aligned}
F_3&\in\mathbb{R}^{B\times192\times80\times80}\\
F_4&\in\mathbb{R}^{B\times192\times40\times40}\\
F_5&\in\mathbb{R}^{B\times192\times20\times20}
\end{aligned}
$$

当前配置的 use_encoder_idx=[2]，表示只对第三个尺度，也就是 1/32 特征执行 Transformer Encoder。该尺度会被展平为 Token 序列，并加入 2D Sin-Cos Position Embedding：

$$
X_5\in\mathbb{R}^{B\times400\times192}
$$

经过 Transformer Encoder 后再还原为二维特征图。

随后执行：

- **Top-down FPN**：从高层向低层逐级上采样并融合。
- **Bottom-up PAN**：从低层向高层逐级下采样并融合。

DEIMv2 的 ViT 配置使用 fuse_op=sum，即同尺度特征相加，而不是通道拼接。version=deim 对应代码中的 RepNCSPELAN5 融合模块。

Nano 的 HybridEncoder 使用两尺度输入，并只在第二个尺度上进行 Transformer Encoder 交互。

### 4.4 为什么要分成 HybridEncoder 和 LiteEncoder？

| 比较 | HybridEncoder | LiteEncoder |
|---|---|---|
| 典型模型 | S/M/L/X、Nano | Pico/Femto/Atto |
| 输入尺度 | 2 或 3 个 | 1 个 |
| Transformer Encoder | 有，作用于指定尺度 | 无 |
| 多尺度融合 | FPN + PAN | 下采样 + GAP Fusion + FPN/PAN |
| 主要目标 | 保留较强特征表达能力 | 降低参数量和计算量 |

## 五、DEIM Transformer Decoder：从候选位置到最终检测框

Encoder 输出多尺度特征后，Decoder 需要从中选出一组候选位置，并逐层细化它们。

### 5.1 多尺度特征展平

以 S 模型为例，输入 Decoder 的特征图尺寸为：

- 1/8：80×80，共 6400 个位置。
- 1/16：40×40，共 1600 个位置。
- 1/32：20×20，共 400 个位置。

合计：

$$
L=80^2+40^2+20^2=8400
$$

经过通道投影和展平后：

$$
Memory\in\mathbb{R}^{B\times8400\times192}
$$

Nano/Pico/Femto/Atto 使用两尺度输入，Token 数量和隐藏维度则根据配置变化。

### 5.2 Anchor Generation：为每个空间位置生成初始参考框

DEIMv2 沿用 RT-DETR/D-FINE 一类的 Anchor 初始化思路。

对每个特征层的空间位置 $(x,y)$，生成归一化中心坐标：

$$
c_x=\frac{x+0.5}{W},\qquad
c_y=\frac{y+0.5}{H}
$$

初始宽高根据尺度设置：

$$
w_l=h_l=0.05\times2^l
$$

其中 $l$ 表示特征层级。

将 Anchor 转换到未激活空间，供后续 MLP 进行预测修正。超出有效范围的 Anchor 会通过 valid_mask 排除。

### 5.3 Query Selection：如何选出 Decoder 的初始 Query？

每个 Encoder 空间位置都经过分类 Head，得到类别分数：

$$
S_{\text{enc}}\in\mathbb{R}^{B\times L\times C}
$$

默认 query_select_method=default，代码对每个空间位置取最大类别分数，再选取 Top-K：

$$
I_{\text{topk}}=
\operatorname{TopK}\left(\max_c S_{\text{enc}}(l,c)\right)
$$

对于选中的位置：

1. 提取对应的 Encoder 特征，作为 Query Content。
2. 使用 Encoder Box Head 对 Anchor 进行修正，形成初始参考框。
3. 将 Query Content 和参考框送入 Decoder。

默认 learn_query_content=False，因此 Query Content 来自被选中的 Encoder 特征，而不是单独的一组随机或固定可学习 Query Embedding。

以 S 模型为例：

$$
Q_{\text{content}}\in\mathbb{R}^{B\times300\times192}
$$

$$
R_{\text{init}}\in\mathbb{R}^{B\times300\times4}
$$

这一步使 Decoder 一开始就获得了具有一定分类置信度和空间定位信息的候选目标。

### 5.4 Denoising Query：训练时额外加入带噪声的 GT

DEIMv2 的配置包含：

- num_denoising=100
- label_noise_ratio=0.5
- box_noise_scale=1.0

训练阶段，从 GT 类别和框生成带噪声的正、负去噪 Query：

$$
GT\xrightarrow{\text{加噪}}Noisy\ GT
$$

Decoder 需要从带噪声的类别和位置中恢复真实 GT。

代码通过 Attention Mask 限制普通匹配 Query 与去噪 Query 之间的信息泄漏，并对不同去噪组之间进行隔离。

这里的 num_denoising=100 是配置参数，实际生成的去噪 Query 数量会根据一个 Batch 中的最大 GT 数量和分组数计算，不应简单理解为每张图片永远额外增加 100 个 Query。

**Denoising Query 只在训练时使用，推理时不会加入。**

### 5.5 Decoder Layer 的内部结构

DEIMv2 的 Decoder Layer 主要包含：

1. Query Self-Attention。
2. Multi-Scale Deformable Cross-Attention。
3. Gateway 或普通残差融合。
4. SwiGLU FFN。
5. RMSNorm。

~~~mermaid
flowchart TD
    A["Query + Query Position Embedding"] --> B["Self-Attention"]
    B --> C["Residual + RMSNorm"]
    C --> D["Multi-Scale Deformable Cross-Attention"]
    E["Encoder Multi-scale Memory"] --> D
    F["Reference Points"] --> D
    D --> G["Gateway 或 Residual Fusion"]
    G --> H["SwiGLU FFN"]
    H --> I["Residual + RMSNorm"]
    I --> J["更新后的 Query"]
~~~

#### 5.5.1 Self-Attention

Self-Attention 让不同 Object Query 之间进行信息交互。

假设 Query 数量为 300：

$$
Q,K,V\in\mathbb{R}^{B\times300\times D}
$$

其注意力计算形式为：

$$
Attention(Q,K,V)=
Softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

注意力矩阵形状为：

$$
A\in\mathbb{R}^{B\times300\times300}
$$

它使不同候选目标之间能够交换信息，帮助模型建模目标间的关系。

#### 5.5.2 Multi-Scale Deformable Cross-Attention

标准全局 Cross-Attention 需要让每个 Query 与所有图像位置进行注意力计算，成本较高。

DEIMv2 使用多尺度可变形注意力。每个 Query 根据参考点预测采样偏移量和注意力权重，只在多尺度特征图上采样少量位置。

以 ViT 路线的配置为例：

~~~yaml
num_points: [3, 6, 3]
num_levels: 3
nhead: 8
~~~

表示每个 Attention Head 在三个特征尺度上分别使用 3、6、3 个采样点。

对于一个 Query，每个 Head 总共采样：

$$
3+6+3=12
$$

个位置，而不是遍历所有 8400 个图像 Token。

采样位置由 Query 和 Reference Point 共同决定，采样特征再按照学习得到的注意力权重进行加权融合。

这也是 DETR 类模型能够在保持全局多尺度信息访问能力的同时控制计算量的重要原因。

#### 5.5.3 Gateway：如何融合 Cross-Attention 的输出？

DEIMv2 的 Decoder 支持 use_gateway。

Gateway 根据当前 Query 和 Cross-Attention 输出生成两个门控权重：

$$
[g_1,g_2]=
\sigma(W[x_1;x_2]+b)
$$

然后融合：

$$
y=\operatorname{RMSNorm}(g_1\odot x_1+g_2\odot x_2)
$$

其中：

- $x_1$：当前 Query 特征。
- $x_2$：Cross-Attention 输出。
- $g_1,g_2$：学习得到的门控系数。

如果关闭 Gateway，则使用普通残差相加和 RMSNorm。

S/M/L/X 和 Nano 使用默认 Gateway 配置；Pico/Femto/Atto 在对应 YAML 中设置 use_gateway=False。

#### 5.5.4 SwiGLU FFN

DEIMv2 将传统 FFN 替换为 SwiGLU 形式：

$$
\begin{aligned}
[u,v]&=W_{12}x\\
h&=\operatorname{SiLU}(u)\odot v\\
FFN(x)&=W_3h
\end{aligned}
$$

它通过两个并行线性投影和门控乘法增强非线性表达能力。

官方代码中的 SwiGLUFFN 使用一个 Linear 将输入映射到两倍隐藏维度，再切分为 $u,v$，经过 SiLU 和逐元素乘法后映射回模型维度。

#### 5.5.5 RMSNorm

RMSNorm 不像 LayerNorm 那样先减去均值，而是按均方根进行归一化：

$$
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\operatorname{Mean}(x^2)+\epsilon}}\odot\gamma
$$

它减少了归一化计算中的一部分操作。DEIMv2 将 RMSNorm 用于 Decoder 的多个残差子层。

#### 5.5.6 共享 Query Position Embedding

Decoder 根据初始参考框计算 Query Position Embedding：

$$
P_q=MLP(R_{\text{init}})
$$

论文指出，迭代过程中 Query 的位置变化相对有限，因此在 Decoder 多层之间共享同一个 Query Position Embedding，而不是每层都重新计算一套。

这减少了重复计算。需要注意，参考框本身仍然会逐层更新；共享的是位置嵌入，不是固定所有层的预测框。

### 5.6 D-FINE 风格的 Fine-Grained Distribution Refinement

DEIMv2 的边界框回归不是简单地让 MLP 直接输出四个连续坐标，而是预测四条边的离散分布。

对于每个 Query，预测：

- 左边界距离 $l$。
- 上边界距离 $t$。
- 右边界距离 $r$。
- 下边界距离 $b$。

配置中：

~~~yaml
reg_max: 32
reg_scale: 4
~~~

每条边对应 reg_max+1=33 个离散位置，因此每个 Query 的边界分布输出维度为：

$$
4\times33=132
$$

对每条边的预测 logits 做 Softmax：

$$
p_n=\operatorname{Softmax}(z)_n
$$

然后使用非均匀 Weighting Function $W(n)$ 将离散分布积分为连续距离：

$$
\hat d=\sum_{n=0}^{32}p_nW(n)
$$

与均匀分桶不同，D-FINE 的 Weighting Function 使用非均匀分布，能够对中心附近和较大偏移范围进行不同粒度的表示。

四个距离再结合 Reference Box 解码为 Bounding Box。

Decoder 的每一层都会产生新的边界分布和预测框，并使用预测结果更新后续层的参考位置，形成迭代细化过程。

### 5.7 LQE：利用边界分布估计定位质量

官方 Decoder 还包含 Location Quality Estimator（LQE）。

它会从四条边的预测分布中提取概率统计信息，例如每条边概率最大的若干个 Bin 及其均值，再通过小型 MLP 得到质量分数。

最终将质量分数加到分类 logits：

$$
S_{\text{final}}=S_{\text{cls}}+S_{\text{quality}}
$$

因此，模型的分类置信度不只来自类别特征，还会参考边界分布所表达的定位质量。

### 5.8 Decoder 的完整输出

对于 S/M/L/X，训练时通常使用 300 个 Query。Decoder 的最终输出包括：

$$
Pred_{\text{logits}}\in\mathbb{R}^{B\times300\times C}
$$

$$
Pred_{\text{boxes}}\in\mathbb{R}^{B\times300\times4}
$$

训练阶段还会返回：

- 中间 Decoder 层辅助输出。
- Encoder Top-K 辅助输出。
- 初始预测输出。
- Denoising Query 输出。
- FDR 相关的边界分布和参考点信息。

这些额外输出用于训练损失计算；推理阶段不需要全部保留。

## 六、训练机制：Dense O2O、MAL 与 Copy-Blend

DEIMv2 的训练设计很大程度上继承自 DEIM。理解这部分需要先明确 DETR 的一对一匹配。

### 6.1 为什么要改进 One-to-One Matching？

原始 DETR 通过 Hungarian Matching，将预测集合与 GT 集合进行一对一匹配：

$$
1\ GT\leftrightarrow1\ Prediction
$$

这保证了一个 GT 不会在训练目标中被多个预测重复负责，是端到端集合预测的重要组成部分。

但一对一匹配也带来监督稀疏的问题：每张图像中的 GT 数量有限，而 Query 数量通常较多，大部分 Query 都是未匹配的背景预测。

DEIM 提出 Dense O2O：

**不是让一个 GT 匹配多个 Query，而是通过数据增强让一张训练图像中包含更多 GT。**

例如，原图有 10 个目标，经过 Mosaic 后可能包含来自多个图像的目标。Hungarian Matching 仍然在增强后的图像上执行一对一匹配，但有效正样本数量增加了。

这与 One-to-Many Matching 是不同概念。

### 6.2 MAL：Matchability-Aware Loss

Dense O2O 增加了匹配数量，但也会引入更多低质量匹配。如果分类损失只把所有匹配样本都当作同样质量的正样本，就可能无法合理处理这些低质量匹配。

MAL 将分类监督与匹配质量联系起来。

令：

- $y$：类别 One-Hot 标签。
- $IoU$：当前预测框与匹配 GT 的 IoU。
- $p=\sigma(z)$：模型预测的分类概率。
- $\gamma$：控制质量分数的幂次。

对于匹配样本，目标分数为：

$$
t= y\cdot IoU^\gamma
$$

对于未匹配类别，目标分数为 0。

官方实现中，当 mal_alpha 未设置时，权重可以概括为：

$$
w=p^\gamma(1-y)+y
$$

最终使用带权重的 Binary Cross Entropy：

$$
L_{\text{MAL}}=
\operatorname{BCEWithLogits}(z,t;w)
$$

这里的关键是：

- 对匹配的正类别，分类目标与定位质量 IoU 相关。
- 对未匹配类别，权重与预测置信度相关。
- 低质量匹配不再被简单当作与高质量匹配完全相同的正样本。

当前基础配置中：

~~~yaml
DEIMCriterion:
  gamma: 1.5
  alpha: 0.75
~~~

需要注意：在当前 loss_labels_mal 实现中，MAL 使用的是 gamma；不要将配置中的 alpha 误解成 MAL 公式中必然存在的系数。

### 6.3 Matcher：训练后期切换匹配代价

DEIMv2 默认仍然使用 Hungarian Matcher。

初始阶段的匹配代价由分类、L1 框距离和 GIoU 共同构成：

$$
C=
\lambda_{\text{cls}}C_{\text{cls}}
+\lambda_{\text{bbox}}C_{\text{bbox}}
+\lambda_{\text{giou}}C_{\text{giou}}
$$

官方配置还提供一个后期切换机制。到达 matcher_change_epoch 后，匹配代价改为：

$$
C_{\text{new}}=-p_{\text{cls}}\cdot IoU^\alpha
$$

当前常见配置为：

~~~yaml
change_matcher: True
iou_order_alpha: 4.0
~~~

这意味着后期匹配更加重视分类分数和定位 IoU 的联合质量。不同规格的 matcher_change_epoch 不相同，应以具体模型 YAML 为准，不能把某个规格的值直接套用到所有模型。

### 6.4 FGL：Fine-Grained Localization Loss

FGL 来自 D-FINE 的分布式定位思想。

与直接对四个坐标计算回归误差不同，FGL 监督四条边的离散分布。

首先根据 GT 框和 Reference Box 计算四个方向的目标距离：

$$
d=(l,t,r,b)
$$

然后将连续距离映射到相邻的两个离散 Bin，并根据目标距离在两个 Bin 之间的位置计算插值权重。

对于每条边，使用相邻 Bin 的交叉熵加权计算分布损失，并使用预测框与 GT 的 IoU 调节样本权重。

这样模型学习的不只是一个最终坐标，而是边界位置的概率分布。

### 6.5 DDF：Decoupled Distillation Focal Loss

DDF 也是从 D-FINE 相关设计继承而来。

这里的 Teacher 不是 DINOv3，而是 Decoder 更深层或最终层产生的定位分布。

其主要过程：

1. 最终 Decoder 层产生定位分布，作为较深层的教师信号。
2. 其他 Decoder 层产生自己的定位分布。
3. 使用带温度系数的 KL Divergence，让不同层之间进行定位分布蒸馏。
4. 根据匹配情况和预测质量调整正、负样本的蒸馏权重。

可以概括为：

$$
L_{\text{DDF}}
\propto
T^2 D_{\text{KL}}
\left(
P_{\text{teacher}}^{T}
\parallel
P_{\text{student}}^{T}
\right)
$$

官方实现的温度参数为 T=5。

**这里必须区分两种蒸馏：**

- ViT-Tiny/Tiny+ 的 Backbone 蒸馏：从 DINOv3 ViT-Small 学习视觉特征。
- DDF：检测 Decoder 内部不同层之间的定位分布自蒸馏。

它们发生在不同模块、解决不同问题。

### 6.6 完整损失函数

论文给出的总损失由五部分组成：

$$
\begin{aligned}
L_{\text{total}}={}&
\lambda_1L_{\text{MAL}}
+\lambda_2L_{\text{FGL}}\\
&+\lambda_3L_{\text{DDF}}
+\lambda_4L_{\text{bbox}}
+\lambda_5L_{\text{GIoU}}
\end{aligned}
$$

对应权重：

| Loss | 权重 | 主要作用 |
|---|---:|---|
| MAL | 1.0 | 匹配质量感知的分类监督 |
| FGL | 0.15 | 细粒度边界分布定位 |
| DDF | 1.5 | Decoder 定位分布自蒸馏 |
| L1 BBox | 5.0 | 边界框坐标回归 |
| GIoU | 2.0 | 优化预测框与 GT 的几何重叠关系 |

官方代码将 FGL 和 DDF 组合在 local Loss 中。

对于 Pico、Femto、Atto，作者发现加入 FGL/DDF 会降低性能，因此这些规格只使用 MAL、L1 和 GIoU，不使用 local Loss。Nano、S、M、L、X 使用 Local Loss。

### 6.7 多层辅助监督与匹配集合

训练阶段并不只对最后一个 Decoder 输出计算损失。

官方 Criterion 还会处理：

- 各 Decoder 中间层的辅助输出。
- Encoder Top-K 候选输出。
- 初始预测输出。
- Denoising Query 输出。

当前实现中，use_uni_set=True 时，Box 和 Local Loss 可以使用跨 Decoder 层、Encoder 辅助输出的匹配并集；分类 MAL 则按对应输出的匹配结果计算。

这有助于让中间层也获得定位监督，而不是只依赖最后一层。

## 七、数据增强：Mosaic、MixUp 与 Copy-Blend

DEIMv2 继承 DEIM 的图像级增强，并在此基础上增加对象级 Copy-Blend。

### 7.1 Mosaic

Mosaic 将多张图片组合成一张训练图像，使同一张图像中出现更多目标和不同尺度的目标。

在 DEIMv2 中，Mosaic 通常在 Dataset Transform 阶段执行，输出尺寸一般为最终输入尺寸的一半，再经过后续 Resize。

例如最终输入 640×640，Mosaic 的 output_size 通常设为 320。

### 7.2 MixUp

MixUp 将两张图像按照比例混合，同时合并两张图像对应的类别和 Bounding Box：

$$
I_{\text{mix}}=\beta I_1+(1-\beta)I_2
$$

其中 $\beta$ 在接近 0.5 的范围内随机选择。

### 7.3 Copy-Blend

Copy-Blend 是 DEIMv2 在对象级别进一步增加监督的方式。

它从 Batch 中收集满足面积条件的目标，根据 GT Bounding Box 裁剪对象区域，然后随机选择位置，将对象区域混合到其他图像中，并同步添加对应的类别和 Bounding Box 标签。

~~~mermaid
flowchart TD
    A["一个 Batch 的图片和 GT"] --> B["收集有效 GT Box"]
    B --> C["按面积阈值过滤"]
    C --> D["随机选取若干对象"]
    D --> E["按 Bounding Box 裁剪图像 Patch"]
    E --> F["随机选择目标图像与位置"]
    F --> G["Copy-Blend 融合"]
    G --> H["更新图像"]
    G --> I["更新 Bounding Box、类别和面积"]
    H --> J["增强后的 Batch"]
    I --> J
~~~

融合可以写成：

$$
I_{\text{new}}=
\beta I_{\text{target}}+
(1-\beta)I_{\text{patch}}
$$

默认配置中常见参数：

~~~yaml
copyblend_prob: 0.5
area_threshold: 100
num_objects: 3
with_expand: True
expand_ratios: [0.1, 0.25]
~~~

含义：

- copyblend_prob：启用 Copy-Blend 的概率。
- area_threshold：筛选对象的面积阈值。
- num_objects：每张图像最多添加的对象数量。
- with_expand：是否在裁剪区域周围扩展上下文。
- expand_ratios：扩展比例范围。

**论文与源码之间有一个值得注意的细节：** 论文文字将 Copy-Blend 描述为添加不带背景的对象；但在当前公开实现中，源码根据 GT Bounding Box 裁剪矩形 Patch，并没有使用实例分割 Mask 做前景抠图。因此，实际复制的 Patch 可能包含框内背景或周围上下文。论文描述的是方法目标，源码展示的是可复现的具体操作，两者不应被混为一谈。

### 7.4 MixUp 与 Copy-Blend 是否同时执行？

当前 BatchImageCollateFunction.apply_mixup() 使用 if ... elif：

- 如果本次 Batch 触发 MixUp，则执行 MixUp。
- 否则才有机会执行 Copy-Blend。

因此二者不是每个 Batch 都必然同时执行的增强。

### 7.5 各规格的增强配置

以下数据来自论文 Table 2：

| 规格 | 输入尺寸 | 论文 Epoch | Mosaic 概率 | MixUp 概率 | Copy-Blend 概率 | Local Loss |
|---|---:|---:|---:|---:|---:|---|
| X | 640 | 50 | 0.5 | 0.5 | 0.5 | 是 |
| L | 640 | 60 | 0.5 | 0.5 | 0.5 | 是 |
| M | 640 | 90 | 0.5 | 0.5 | 0.5 | 是 |
| S | 640 | 120 | 0.5 | 0.5 | 0.5 | 是 |
| Nano | 640 | 148 | 0.5 | 0.5 | 0.4 | 是 |
| Pico | 640 | 468 | 0.5 | 0 | 0 | 否 |
| Femto | 416 | 468 | 0.5 | 0 | 0 | 否 |
| Atto | 320 | 468 | 0.3 | 0 | 0 | 否 |

注意：论文 Table 2 的 Epoch 与当前 YAML 中的 epoches 配置字段值并不总是相同。官方配置会额外包含 EMA 搜索或训练尾段；例如 S 的 YAML 为 132，而论文表中报告 120。复现实验时必须使用对应规格的完整 YAML，不能只复制论文表格中的 Epoch 数。

### 7.6 优化器与学习率

论文 Table 2 使用 AdamW。不同 Backbone 路线的学习率并不相同：预训练 ViT Backbone 使用较小的学习率，检测 Encoder/Decoder 使用较大的学习率；HGNetv2 轻量模型则采用更大的整体学习率。

| 规格 | Optimizer | Base LR | Min LR | Backbone LR | Weight Decay |
|---|---|---:|---:|---:|---:|
| X | AdamW | 5e-4 | 2.5e-4 | 1e-6 | 1.25e-4 |
| L | AdamW | 5e-4 | 2.5e-4 | 1.25e-5 | 1.25e-4 |
| M | AdamW | 5e-4 | 2.5e-4 | 2.5e-5 | 1e-4 |
| S | AdamW | 5e-4 | 2.5e-4 | 2.5e-5 | 1e-4 |
| Nano | AdamW | 8e-4 | 8e-4 | 4e-4 | 1e-4 |
| Pico | AdamW | 1.6e-3 | 8e-4 | 8e-4 | 1e-4 |
| Femto | AdamW | 1.6e-3 | 8e-4 | 8e-4 | 1e-4 |
| Atto | AdamW | 2e-3 | 1e-3 | 1e-3 | 1e-4 |

这些是论文表格中的训练超参数，不应直接当作自定义数据集的最优配置。尤其当目标数据集规模、Batch Size、图像尺寸或预训练权重发生变化时，需要重新验证学习率和训练稳定性。

## 八、训练与推理的数据流

### 8.1 完整训练流程

~~~mermaid
flowchart TD
    A["COCO 格式图片与 GT"] --> B["Mosaic / 随机裁剪 / 翻转"]
    B --> C["Batch Collate"]
    C --> D{"增强选择"}
    D -->|"满足 MixUp 条件"| E["MixUp"]
    D -->|"未执行 MixUp，满足条件"| F["Copy-Blend"]
    D -->|"都未触发"| G["保持当前 Batch"]
    E --> H["Backbone"]
    F --> H
    G --> H
    H --> I["Encoder"]
    I --> J["Query Selection"]
    J --> K["Decoder + Denoising Queries"]
    K --> L["分类、框与分布输出"]
    L --> M["Hungarian Matcher"]
    M --> N["MAL / L1 / GIoU / FGL / DDF"]
    N --> O["辅助层和 Encoder Loss"]
    O --> P["总 Loss"]
    P --> Q["Backpropagation"]
    Q --> R["Optimizer Update + EMA"]
~~~

训练过程中的 Denoising Query、辅助输出和跨层匹配集合，都是为了增加监督信号；推理时不需要这些训练辅助分支。

### 8.2 推理阶段的输出

模型最终输出：

$$
Pred_{\text{logits}}\in\mathbb{R}^{B\times Q\times C}
$$

$$
Pred_{\text{boxes}}\in\mathbb{R}^{B\times Q\times4}
$$

其中 Box 使用归一化的 $(c_x,c_y,w,h)$ 格式。

### 8.3 PostProcessor 如何得到检测结果？

官方 PostProcessor 主要执行以下步骤：

1. 将归一化的中心点框转换为角点框 $(x_1,y_1,x_2,y_2)$。
2. 根据原图尺寸将坐标缩放到像素坐标。
3. 对分类 logits 使用 Sigmoid。
4. 将 Query 与类别维度展平，执行 Top-K。
5. 根据展平索引恢复类别 ID 和 Query ID。
6. 根据 Query ID 取回对应 Bounding Box。
7. 返回 labels、boxes 和 scores。

由于采用 Focal Loss 分支，分类分数是 Sigmoid，而不是 Softmax。

对于一个图像，假设 $Q=300$、类别数 $C=80$，展平后的候选分数数量为：

$$
300\times80=24000
$$

默认 num_top_queries=300，从这些 Query-Class 组合中选取 Top-K。

**这不是 NMS。** PostProcessor 进行的是分数 Top-K 筛选，没有执行传统检测器中的 Non-Maximum Suppression。DETR 的集合预测与一对一匹配设计，正是为了减少重复预测并支持端到端检测。

### 8.4 训练与推理的差异

| 项目 | 训练阶段 | 推理阶段 |
|---|---|---|
| Denoising Query | 使用 | 不使用 |
| Encoder 辅助输出 | 计算辅助 Loss | 不需要 |
| Decoder 中间输出 | 用于辅助监督 | 只保留配置指定的预测层 |
| Matcher | Hungarian Matching | 不使用 |
| Loss | MAL、BBox、GIoU、FGL、DDF 等 | 不计算 |
| PostProcessor | 验证时使用 | 使用 |
| NMS | 不使用 | 不使用 |

## 九、实验结果与性能分析

### 9.1 COCO 主结果

论文使用 COCO 2017 验证集报告主要结果。模型精度采用 COCO AP，即 IoU 从 0.50 到 0.95、步长 0.05 的平均 AP。

主要模型结果如下：

| 模型 | 参数量 | GFLOPs | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEIMv2-S | 9.71M | 49.5 | 50.9 | 68.3 | 55.1 | 31.4 | 55.3 | 70.3 |
| DEIMv2-M | 18.11M | 84.1 | 53.0 | 70.2 | 57.6 | 34.2 | 57.4 | 71.5 |
| DEIMv2-L | 32.18M | 144.5 | 56.0 | 73.4 | 60.9 | 37.5 | 60.8 | 75.2 |
| DEIMv2-X | 50.26M | 200.0 | 57.8 | 75.4 | 63.2 | 39.2 | 62.9 | 75.9 |

轻量模型：

| 模型 | 参数量 | GFLOPs | 输入尺寸 | AP |
|---|---:|---:|---:|---:|
| Nano | 3.57M | 6.86 | 640 | 43.0 |
| Pico | 1.51M | 5.15 | 640 | 38.5 |
| Femto | 0.96M | 1.67 | 416 | 31.0 |
| Atto | 0.49M | 0.76 | 320 | 23.8 |

### 9.2 与 DEIM 的对比

将 DEIMv2 与前代 DEIM 在相近模型规模下进行对比，可以看到：

| 规格 | DEIM AP | DEIMv2 AP | DEIM 参数量 | DEIMv2 参数量 | DEIM GFLOPs | DEIMv2 GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| S | 49.0 | 50.9 | 约 10M | 9.71M | 25 | 49.5 |
| M | 52.7 | 53.0 | 约 19M | 18.11M | 57 | 84.1 |
| L | 54.7 | 56.0 | 约 31M | 32.18M | 91 | 144.5 |
| X | 56.5 | 57.8 | 约 62M | 50.26M | 202 | 200.0 |

这组数据说明：

1. DINOv3 路线整体提高了模型精度，尤其在 L/X 规格中表现明显。
2. 参数量并不一定增加，X 的参数量反而从约 62M 降至约 50M。
3. 计算量并非所有规格都下降。S/M/L 的 GFLOPs 明显高于对应 DEIM，而 X 的计算量相近。
4. 因此不能简单说 DEIMv2 在所有模型规模上都比 DEIM 更省算力。它改善的是精度、参数量和计算量之间的整体折中。

### 9.3 DINOv3 带来的提升主要来自哪里？

论文特别分析了不同目标尺寸上的结果。

例如：

| 模型 | DEIM AP_S | DEIMv2 AP_S | DEIM AP_M | DEIMv2 AP_M | DEIM AP_L | DEIMv2 AP_L |
|---|---:|---:|---:|---:|---:|---:|
| S | 30.4 | 31.4 | 52.6 | 55.3 | 65.7 | 70.3 |
| X | 38.8 | 39.2 | 61.4 | 62.9 | 74.2 | 75.9 |

可以看到：

- 中目标和大目标的提升更明显。
- 小目标 AP 的提升相对有限。
- DINOv3 提供的强语义特征，并没有自动解决小目标细节不足的问题。

这点对密集小目标检测尤其重要。STA 的 CNN 细节路径有助于补充局部信息，但论文结果并不支持“接入 DINOv3 后小目标自然大幅提升”的结论。

### 9.4 轻量模型的价值

DEIMv2 的轻量路线与 DINOv3 路线目标不同：

- Pico：1.51M 参数、5.15 GFLOPs、38.5 AP。
- Femto：0.96M 参数、1.67 GFLOPs、31.0 AP。
- Atto：0.49M 参数、0.76 GFLOPs、23.8 AP。

它们不依赖 DINOv3，而是通过裁剪 HGNetv2、缩小 Encoder/Decoder 维度和减少 Query 数量来控制资源消耗。

因此，选择模型时应结合实际设备的算力、内存和目标精度要求，而不是只看模型名字中的 S/M/L/X。

### 9.5 FLOPs 与延迟数据的版本问题

官方仓库在 2026-10-22 更新说明中指出，旧版 FLOPs 计算工具遗漏了 Attention FLOPs，之后修正了统计逻辑。

因此，本文采用修正后的 GFLOPs：

- S：49.5G。
- M：84.1G。
- L：144.5G。
- X：200.0G。

旧资料中出现的 S 25.6G、M 52.2G、L 96.7G、X 151.6G 等数值属于旧统计口径，不应与修正后的数据混用。

另外，论文 Table 1 与当前 README 对 Atto/Femto 的延迟数据存在差异。例如论文表格中 Atto/Femto 为 1.61/1.91 ms，而当前 README 为 1.10/1.45 ms。由于这些数据需要结合具体硬件、精度、推理框架和测试流程才能进行公平比较，本文不将延迟作为统一横向排名依据。实际部署应在目标设备上重新 Benchmark。

### 9.6 论文没有充分回答的问题

论文 v5 提供了模型规格、训练配置和不同检测器的对比结果，但没有给出一套能够逐一隔离 STA、Copy-Blend、SwiGLU、RMSNorm 等组件贡献的完整消融表。

因此：

- 可以确认作者提出了这些设计。
- 可以确认完整 DEIMv2 模型取得了论文报告的结果。
- 但不能仅凭总结果断言每个模块分别贡献了多少 AP。
- 如果要判断 STA 或 Copy-Blend 对某个具体数据集是否有效，需要自己做控制变量实验。

## 十、模型局限性与实际应用分析

### 10.1 小目标并不是 DINOv3 路线的明显优势

DINOv3 的 ViT Backbone 使用 Patch Embedding，原始 Patch Size 为 16。对于 640×640 输入，原始 ViT Patch 网格是 40×40。

STA 将语义特征上采样到 1/8，并通过 CNN 细节分支补充信息，但这不等于 Backbone 原生就具有 1/8 的 ViT Token 分辨率。

对于特别小的目标：

- 目标可能只覆盖很少的输入像素。
- Patch Embedding 会将局部像素聚合为 Patch Token。
- 后续上采样只能调整特征图尺寸，无法恢复已经丢失的全部细节。
- CNN 细节分支能提供补充，但是否足够仍需要实验验证。

因此，密集小目标场景应该重点观察 AP_S、召回率、漏检率和相邻目标分离能力，而不应只看总体 AP。

### 10.2 高分辨率不等于一定提升

DEIMv2 默认大部分模型使用 640 输入。提高输入尺寸会增加特征图空间位置数量，也会增加注意力、Encoder 和 Decoder 的计算负担。

如果要使用 960 或 1280 输入，需要同步调整：

- eval_spatial_size。
- Train/Val Resize。
- Mosaic 的 output_size。
- Batch Size 和显存配置。
- Anchor 与多尺度特征缓存相关设置。

不能只修改模型输入尺寸而忽略训练数据管线和部署输入。

### 10.3 ViT 路线与 HGNetv2 路线如何选择？

| 场景 | 可以优先测试的路线 | 需要关注 |
|---|---|---|
| 高精度、算力充足 | S/M/L/X | AP、GFLOPs、显存和延迟 |
| 边缘设备、算力较低 | Nano/Pico | 精度与速度折中 |
| 极低算力、移动端 | Femto/Atto | 输入尺寸和目标尺寸限制 |
| 极小目标、密集目标 | 不应直接根据模型规模选择 | AP_S、输入分辨率、有效特征尺度、漏检率 |

对于密集植物或其他小目标任务，建议先做小规模对照实验：

1. 使用相同数据划分和标注，比较 DINOv3/STA 与 HGNetv2 路线。
2. 固定训练策略，分别比较 640、960 等输入尺寸。
3. 单独统计小目标 AP、召回率和相邻目标的重复/漏检情况。
4. 观察高分辨率是否带来真实收益，而不只比较总体 AP。

## 十一、官方源码、配置与复现

### 11.1 重要源码文件

本文核对的官方代码版本：

[f005f50ce565e63f6c298feccb51901a98c3d035](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/f005f50ce565e63f6c298feccb51901a98c3d035)

| 文件 | 作用 |
|---|---|
| engine/deim/deim.py | 主模型前向：Backbone → Encoder → Decoder |
| engine/backbone/dinov3_adapter.py | DINOv3STAs、SpatialPriorModulev2 和 STA 特征融合 |
| engine/backbone/vit_tiny.py | ViT-Tiny/Tiny+ 实现 |
| engine/deim/hybrid_encoder.py | HybridEncoder、FPN/PAN 融合 |
| engine/deim/lite_encoder.py | 轻量级 LiteEncoder |
| engine/deim/deim_decoder.py | DEIMTransformer、Query Selection、Decoder 和 FDR |
| engine/deim/dfine_decoder.py | MSDeformableAttention、Integral、LQE 等定位组件 |
| engine/deim/deim_criterion.py | MAL、BBox、GIoU、FGL、DDF 等损失 |
| engine/deim/matcher.py | Hungarian Matcher 和后期匹配代价切换 |
| engine/deim/denoising.py | Contrastive Denoising Query 构造 |
| engine/data/dataloader.py | MixUp、Copy-Blend、Multi-scale Batch Collate |
| engine/deim/postprocessor.py | Top-K、类别恢复和 Bounding Box 解码 |
| tools/deployment/export_onnx.py | ONNX 导出 |

建议阅读顺序：

1. configs/deimv2/：先看模型配置。
2. engine/deim/deim.py：确认主前向路径。
3. dinov3_adapter.py：理解 STA。
4. hybrid_encoder.py 或 lite_encoder.py：理解多尺度特征融合。
5. deim_decoder.py：理解 Query、Deformable Attention 和 FDR。
6. deim_criterion.py、matcher.py：理解训练目标。
7. dataloader.py：理解数据增强。
8. postprocessor.py 和导出脚本：理解推理输出。

### 11.2 配置文件入口

ViT 路线：

~~~text
configs/deimv2/
├── deimv2_dinov3_s_coco.yml
├── deimv2_dinov3_m_coco.yml
├── deimv2_dinov3_l_coco.yml
└── deimv2_dinov3_x_coco.yml
~~~

HGNetv2 路线：

~~~text
configs/deimv2/
├── deimv2_hgnetv2_n_coco.yml
├── deimv2_hgnetv2_pico_coco.yml
├── deimv2_hgnetv2_femto_coco.yml
└── deimv2_hgnetv2_atto_coco.yml
~~~

### 11.3 论文与当前配置的差异：L 模型隐藏维度

论文 v5 的 Table 1 将 L 模型的 Encoder/Decoder 隐藏维度列为 256。

但当前官方 deimv2_dinov3_l_coco.yml 和 Hugging Face L 模型 config.json 中：

~~~yaml
DINOv3STAs:
  hidden_dim: 224

HybridEncoder:
  hidden_dim: 224

DEIMTransformer:
  hidden_dim: 224
~~~

因此，复现当前官方 L 权重时应使用对应的 224 维配置，而不是仅根据论文 Table 1 手工改成 256。否则可能导致结构与 checkpoint 不匹配。

这也说明：复现时必须把论文、官方 YAML 和权重配置对应起来。

### 11.4 预训练 Backbone 与检测权重不是一回事

官方项目提供两类重要权重：

1. **Backbone 预训练权重**：用于初始化 DINOv3 或蒸馏得到的 ViT-Tiny/Tiny+。
2. **DEIMv2 检测模型权重**：已经完成 COCO 检测训练，包含 Backbone、Encoder、Decoder 等完整参数。

S/M 使用的 Tiny/Tiny+ 蒸馏 Backbone 与 L/X 使用的官方 DINOv3 Backbone 不是同一份权重。

官方 README 要求将对应 Backbone 权重放入 ckpts，并根据配置文件中的 weights_path 加载。

### 11.5 Hugging Face 官方模型

作者在 Hugging Face 的 Intellindust 组织下发布了模型。以下为对应入口：

| 规格 | Hugging Face 模型 |
|---|---|
| S | [DEIMv2_DINOv3_S_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO) |
| M | [DEIMv2_DINOv3_M_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO) |
| L | [DEIMv2_DINOv3_L_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO) |
| X | [DEIMv2_DINOv3_X_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO) |
| Nano | [DEIMv2_HGNetv2_N_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO) |
| Pico | [DEIMv2_HGNetv2_PICO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO) |
| Femto | [DEIMv2_HGNetv2_FEMTO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO) |
| Atto | [DEIMv2_HGNetv2_ATTO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO) |

Hugging Face 模型配置文件可以帮助确认具体的隐藏维度、层数、特征尺度和 Query 数。需要注意，作者 Hub 上的模型使用 PyTorchModelHubMixin 方式发布；不要把社区转换权重或其他框架的模型卡直接当成作者原始权重。

## 十二、训练、推理与部署

### 12.1 环境准备

官方 README 推荐 Python 3.11，并说明测试过 PyTorch 2.4.1 / 2.5.1。

~~~bash
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
~~~

建议优先按照官方仓库的依赖文件安装，不要仅凭其他 DETR 项目的环境配置推测兼容性。

### 12.2 自定义数据集

官方代码支持 COCO 格式数据集。

基本目录结构：

~~~text
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
~~~

自定义类别时，需要修改数据集配置：

- 设置正确的 num_classes。
- 将 remap_mscoco_category 设置为 False。
- 修改训练和验证图片目录、标注 JSON 路径。
- 确认类别 ID 连续且与模型分类头定义一致。

ViT 路线的配置包含 ImageNet Mean/Std Normalize；HGNetv2 路线的预处理配置不完全相同。自定义训练时应沿用所选规格 YAML 的预处理方式，不要将两条路线的预处理设置随意混用。

### 12.3 训练命令

ViT 路线，例如 DEIMv2-S：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--use-amp --seed=0
~~~

HGNetv2 路线，例如 DEIMv2-Nano：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_hgnetv2_n_coco.yml \
--use-amp --seed=0
~~~

其中 nproc_per_node 需要根据实际 GPU 数量修改。

### 12.4 测试与微调

测试已有权重：

~~~bash
torchrun --nproc_per_node=4 train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--test-only -r model.pth
~~~

从已有权重继续微调：

~~~bash
torchrun --nproc_per_node=4 train.py \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
--use-amp --seed=0 -t model.pth
~~~

具体参数含义和训练恢复方式以官方 train.py 与 README 为准。

### 12.5 ONNX 导出

官方导出脚本：

~~~bash
pip install onnx onnxsim

python tools/deployment/export_onnx.py \
--check \
-c configs/deimv2/deimv2_dinov3_s_coco.yml \
-r model.pth
~~~

导出脚本将模型转换为 Deploy 模式，并把模型与 PostProcessor 封装在同一个导出模型中。

ONNX 输入：

| 输入 | 形状示意 | 含义 |
|---|---|---|
| images | [N, 3, H, W] | 输入图像 |
| orig_target_sizes | [N, 2] | 原图高度和宽度 |

输出：

- labels
- boxes
- scores

脚本默认使用 ONNX Opset 17，并将 Batch 维度设为动态维度。空间尺寸通常仍由配置文件中的 eval_spatial_size 决定。

### 12.6 TensorRT 部署

官方 README 给出的基本转换方式：

~~~bash
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
~~~

**版本注意事项：**

- 官方建议 FP16 推理使用 TensorRT 10.6 或更高版本。
- README 指出 TensorRT 10.4 的 FP16 推理可能产生错误输出。
- 使用较旧版本时，应考虑 FP32，并且必须对导出的 Engine 做数值和检测精度验证。

不要仅凭 ONNX 导出成功就认为 TensorRT 结果正确。建议使用同一批图片比较 PyTorch、ONNX Runtime 和 TensorRT 的类别、置信度、框坐标与最终 AP。

## 十三、总结：DEIMv2 的真正价值

DEIMv2 的核心不是提出一种完全不同于 DETR 的新检测范式，而是围绕实时检测的精度、计算量和模型规模进行系统设计。

它最值得理解的地方有四个：

1. **DINOv3 + STA**：将自监督 ViT 的高层语义特征与轻量 CNN 的局部空间细节结合，适配多尺度检测。
2. **双 Backbone 路线**：较大模型使用 DINOv3/蒸馏 ViT，极轻量模型使用裁剪 HGNetv2，覆盖不同部署预算。
3. **高效 Decoder 与定位细化**：使用多尺度可变形注意力、SwiGLU、RMSNorm、共享 Query Position Embedding 和 D-FINE 风格分布式定位。
4. **Dense O2O + MAL + Copy-Blend**：从图像级到对象级增加训练监督，并让分类目标感知匹配质量。

但也需要正确理解它的边界：

- DINOv3 带来的精度收益主要体现在中、大目标，小目标提升有限。
- ViT 的 Patch 特征上采样不等于恢复原始细节。
- S/M/L 的计算量并不一定低于同规模的 DEIM CNN 模型。
- 论文没有完整的逐模块消融表，无法将整体提升精确分配到某一个设计。
- 论文、当前 YAML 和 Hub 配置之间存在少量版本差异，复现必须核对具体规格。

如果要深入研究，我建议按照以下顺序阅读代码：

**dinov3_adapter.py → hybrid_encoder.py → deim_decoder.py → deim_criterion.py → dataloader.py → postprocessor.py。**

这条路径能够从图像输入一路追踪到最终检测结果，也能把论文中的 STA、Dense O2O、MAL、FGL、DDF 与实际代码一一对应。

## 参考资料

### 论文

1. [Real-Time Object Detection Meets DINOv3（DEIMv2，arXiv v5）](https://arxiv.org/abs/2509.20787v5)
2. [DEIM: DETR with Improved Matching for Fast Convergence（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html)
3. [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement（ICLR 2025）](https://arxiv.org/abs/2410.13842)
4. [DINOv3（2025）](https://arxiv.org/abs/2508.10104)
5. [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

### 官方代码与模型

1. [DEIMv2 官方仓库](https://github.com/Intellindust-AI-Lab/DEIMv2)
2. [DEIMv2 代码核对版本 f005f50](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/f005f50ce565e63f6c298feccb51901a98c3d035)
3. [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)
4. [DEIMv2 作者 Hugging Face 组织](https://huggingface.co/Intellindust)
5. [DEIMv2 项目页面](https://intellindust-ai-lab.github.io/projects/DEIMv2/)

### 关键源码

- [DINOv3 Adapter / STA](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/backbone/dinov3_adapter.py)
- [ViT-Tiny](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/backbone/vit_tiny.py)
- [HybridEncoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/hybrid_encoder.py)
- [LiteEncoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/lite_encoder.py)
- [DEIM Decoder](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/deim_decoder.py)
- [DEIM Criterion](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/deim_criterion.py)
- [Matcher](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/matcher.py)
- [DataLoader 与 Copy-Blend](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/data/dataloader.py)
- [PostProcessor](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/engine/deim/postprocessor.py)
- [ONNX Export](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/tools/deployment/export_onnx.py)
