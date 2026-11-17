---
title: DEIMv2：融合 DINOv3 的实时 DETR 目标检测器
description: 基于论文、官方 GitHub 源码、模型配置与 Hugging Face 权重，解析 DEIMv2 的整体架构、STA、双 Backbone 路线、Dense O2O、损失函数、训练与部署。
---

# DEIMv2：融合 DINOv3 的实时 DETR 目标检测器

## 一、背景与核心思想

DEIMv2 的论文名称为 **Real-Time Object Detection Meets DINOv3**，作者为 Shihua Huang、Yongjie Hou、Longfei Liu、Xuanlong Yu 和 Xi Shen，发表于 2025 年 9 月的 arXiv。

DEIMv2 的目标不是简单地把 DINOv3 接到一个 DETR 检测头上，而是在实时检测的计算预算下，将 DINOv3 的语义表征能力、DEIM 的 Dense O2O 训练策略，以及面向不同算力预算的轻量化网络设计结合起来。

它主要解决三个问题：

1. **如何把 DINOv3 的强语义特征用于实时检测？** DINOv3 的 ViT 主干天然输出单尺度特征，而检测任务需要多尺度特征。DEIMv2 使用 Spatial Tuning Adapter（STA）将中间层语义特征转换为多尺度特征，并引入轻量级空间先验分支补充局部细节。
2. **如何覆盖不同算力预算？** S、M、L、X 使用 DINOv3 或其蒸馏得到的 ViT-Tiny 系列；Nano、Pico、Femto、Atto 使用经过深度和宽度裁剪的 HGNetv2。
3. **如何进一步提高训练效率和检测效果？** 在 DEIM 的 Dense O2O 与 Matchability-Aware Loss（MAL）基础上，引入对象级 Copy-Blend，并简化 Decoder。

因此，DEIMv2 不应被理解为“DINOv3 + 一个检测头”，它是一套包含 Backbone、特征适配、Hybrid Encoder、Decoder、匹配策略和损失函数的完整实时检测方案。

### 1.1 论文与代码入口

| 资料 | 内容 |
|---|---|
| [论文：Real-Time Object Detection Meets DINOv3](https://arxiv.org/abs/2509.20787) | 方法设计、实验和模型对比 |
| [论文 PDF](https://arxiv.org/pdf/2509.20787) | 完整论文 |
| [官方 GitHub](https://github.com/Intellindust-AI-Lab/DEIMv2) | 源码、配置、训练和部署 |
| [官方项目页面](https://intellindust-ai-lab.github.io/projects/DEIMv2/) | 项目介绍与论文信息 |
| [Hugging Face 官方模型组织](https://huggingface.co/Intellindust) | 预训练模型权重 |
| [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/deimv2) | Transformers 中的 DEIMv2 模型接口 |

本文以论文、官方仓库及其配置和实现为主要依据。论文中的方法描述、官方代码中的实际实现，以及本文为帮助理解而作的解释会分别说明。

## 二、DEIMv2 与 DEIM、RT-DETR、D-FINE 的关系

理解 DEIMv2，首先需要区分几个经常一起出现的名称。

### 2.1 DEIM：重点是训练框架与匹配策略

DEIM 全称为 **DETR with Improved Matching**，论文发表于 CVPR 2025。

DEIM 重点研究 DETR 训练过程中的匹配与监督问题。传统 DETR 使用 One-to-One（O2O）匹配，每个 Ground Truth 通常只与一个预测匹配，正样本监督相对稀疏。

DEIM 提出 Dense O2O，通过 Mosaic、MixUp 等数据增强提高单张训练图像中的有效目标数量，使 O2O 匹配获得更丰富的监督；同时提出 Matchability-Aware Loss（MAL），处理不同匹配质量的样本。

DEIM 可以与 RT-DETR、D-FINE 等检测器结合。因此，DEIM 不应简单等同于某一种固定 Backbone 或完整网络结构。

### 2.2 DEIMv2：在 DEIM 基础上进一步设计完整检测器

DEIMv2 延续 Dense O2O 和 MAL，同时进一步改变了检测器的多个部分：

- S、M、L、X：引入 DINOv3 / 蒸馏 ViT Backbone 和 STA。
- Nano、Pico、Femto、Atto：使用轻量化 HGNetv2。
- Encoder：根据 Backbone 类型使用 HybridEncoder 或 LiteEncoder。
- Decoder：引入 SwiGLU FFN、RMSNorm 和共享 Query Position Embedding。
- 训练：在 Dense O2O 基础上加入对象级 Copy-Blend。

### 2.3 与 RT-DETR、D-FINE 的关系

DEIMv2 的整体结构沿用 RT-DETR 风格的 Backbone、Hybrid Encoder、Decoder 三段式架构；其 Decoder 和边界框分布回归部分也吸收了 D-FINE 的设计。

| 方法 | 主要关注点 | 与 DEIMv2 的关系 |
|---|---|---|
| RT-DETR | 实时端到端 DETR 架构，包含 Hybrid Encoder 和 Transformer Decoder | DEIMv2 沿用相近的整体架构组织方式 |
| D-FINE | 通过 Fine-grained Distribution Refinement 改进边界框回归 | DEIMv2 的 Decoder 和 FGL、DDF 损失使用了相关设计 |
| DEIM | Dense O2O、MAL 和更快的 DETR 训练收敛 | DEIMv2 继承并增强其训练策略 |
| DEIMv2 | DINOv3/STA、轻量 HGNetv2、简化 Decoder、Dense O2O 增强 | 完整的实时检测模型系列 |

需要注意：DEIMv2 并不是 RT-DETR 的简单改名，也不是只给 D-FINE 换了 Backbone。它在 Backbone、特征融合、Decoder 和训练策略上都有明确改动。

## 三、整体网络架构

DEIMv2 的整体结构可以概括为：

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

其中最重要的设计是：**根据模型规格选择不同的 Backbone 和 Encoder 路线，但最终都进入 DEIM Transformer Decoder。**

### 3.1 整体结构图

~~~mermaid
flowchart TD
    A["输入图像"] --> B{"模型规格"}
    B -->|"S / M / L / X"| C["DINOv3 / 蒸馏 ViT-Tiny"]
    C --> D["Spatial Tuning Adapter"]
    A --> E["空间先验分支"]
    E --> D
    D --> F["多尺度特征 P3 / P4 / P5"]
    F --> G["HybridEncoder"]
    B -->|"Nano / Pico / Femto / Atto"| H["HGNetv2 轻量 Backbone"]
    H --> I{"Encoder 类型"}
    I -->|"Nano 等常规轻量配置"| J["HybridEncoder"]
    I -->|"Pico / Femto / Atto"| K["LiteEncoder"]
    J --> L["多尺度检测特征"]
    K --> L
    G --> M["DEIM Transformer Decoder"]
    L --> M
    M --> N["分类与边界框预测"]
    N --> O["Top-K 后处理"]
    O --> P["检测结果"]
~~~

### 3.2 两条 Backbone 路线

DEIMv2 一共提供八种模型规格：

- DINOv3 路线：S、M、L、X。
- HGNetv2 路线：N（论文中也称 Nano）、Pico、Femto、Atto。

两条路线并不是简单地使用同一 Backbone 的不同宽度，而是针对不同算力预算采用了不同设计。

| 规格 | Backbone | 特征适配 | 主要定位 |
|---|---|---|---|
| Atto | HGNetv2 裁剪版 | LiteEncoder | 极低算力、移动端 |
| Femto | HGNetv2 裁剪版 | LiteEncoder | 超轻量 |
| Pico | HGNetv2 裁剪版 | LiteEncoder | 轻量边缘设备 |
| N / Nano | HGNetv2 | HybridEncoder | 轻量实时检测 |
| S | 蒸馏 ViT-Tiny | STA + HybridEncoder | 小模型、高精度 |
| M | 蒸馏 ViT-Tiny+ | STA + HybridEncoder | 中等规模 |
| L | DINOv3 ViT-Small | STA + HybridEncoder | 高精度 |
| X | DINOv3 ViT-Small+ | STA + HybridEncoder | 高性能 |

## 四、DINOv3 Backbone 与 STA

这是 DEIMv2 最值得重点理解的部分。

### 4.1 为什么 DINOv3 不能直接接检测头？

DINOv3 是基于 Vision Transformer 的视觉基础模型。以 ViT 的 Patch Size 为 16 为例，输入图像经过 Patch Embedding 后，空间特征通常对应：

$$
F_{\text{ViT}}\in
\mathbb{R}^{B\times D\times H/16\times W/16}
$$

它主要提供单一空间尺度的密集特征。

但目标检测中的目标尺寸差异很大：

- 小目标需要更高分辨率的特征。
- 中等目标需要中间尺度特征。
- 大目标需要较大感受野和语义信息。

因此，检测网络通常需要类似 P3、P4、P5 的多尺度特征。

如果只使用 DINOv3 最后一层输出，就缺少直接适配检测任务的多尺度信息。

### 4.2 STA 的核心思想

STA 全称为 **Spatial Tuning Adapter**。

它并不是简单地在 DINOv3 后面增加一个普通 FPN，而是包含两条互补的信息路径：

1. **语义特征路径**：从 DINOv3 的多个 Transformer Block 获取中间特征，通过双线性插值调整到不同空间尺度。
2. **空间细节路径**：直接对原始输入图像进行轻量级卷积下采样，提取局部纹理和细节。

最后将两路特征融合，形成检测所需要的多尺度特征。

### 4.3 语义特征路径

DINOv3 ViT 的中间层能够提供不同深度的语义特征。

假设选取三个 Transformer Block：

$$
F_1,F_2,F_3
\in
\mathbb{R}^{B\times D\times H/16\times W/16}
$$

将 Token 转换为二维特征图后，通过双线性插值调整空间大小：

$$
\begin{aligned}
F_{1/8} &= \operatorname{Resize}(F_1,H/8,W/8)\\
F_{1/16} &= \operatorname{Resize}(F_2,H/16,W/16)\\
F_{1/32} &= \operatorname{Resize}(F_3,H/32,W/32)
\end{aligned}
$$

注意：这里的 Resize 不会增加可学习参数。它只是改变特征图的空间尺寸。

论文举例使用 ViT 的第 5、8、11 层特征。具体使用哪些 Block，应以所使用的模型配置和权重版本为准；官方仓库和 Hugging Face 配置可能存在不同版本的层索引设置，不应将不同配置与权重随意混用。

### 4.4 空间先验分支

STA 同时从原始输入图像提取局部细节。

官方代码中的 SpatialPriorModulev2 主要由以下结构组成：

- Stem：卷积、SyncBatchNorm、GELU、MaxPool。
- Conv2：进一步下采样至 1/8。
- Conv3：进一步下采样至 1/16。
- Conv4：进一步下采样至 1/32。

结构如下：

~~~mermaid
flowchart TD
    A["原始图像 B×3×H×W"] --> B["3×3 Conv, stride=2"]
    B --> C["SyncBN + GELU"]
    C --> D["3×3 MaxPool, stride=2"]
    D --> E["1/4 特征"]
    E --> F["3×3 Conv, stride=2"]
    F --> G["1/8 特征"]
    G --> H["3×3 Conv, stride=2"]
    H --> I["1/16 特征"]
    I --> J["3×3 Conv, stride=2"]
    J --> K["1/32 特征"]
~~~

它的作用是补充 DINOv3 中高层语义特征中相对不足的局部空间细节。

例如，田间植物、小目标和边缘细节等任务可能更依赖局部纹理与空间信息，而不只是高层语义。

### 4.5 两路特征融合

对于每一个尺度，将 DINOv3 的语义特征与空间先验特征进行融合。

概念上可以表示为：

$$
F_i=
\operatorname{Conv}_{1\times1}
\left(
\operatorname{Concat}(S_i,P_i)
\right)
$$

其中：

- $S_i$：DINOv3 对应尺度的语义特征。
- $P_i$：空间先验分支对应尺度的局部细节特征。
- $\operatorname{Concat}$：沿通道维度拼接。
- $1\times1$ Conv：将拼接后的通道投影到检测网络需要的维度。

官方实现中还包含 SyncBatchNorm。

完整结构：

~~~mermaid
flowchart TD
    A["输入图像"] --> B["DINOv3 ViT"]
    B --> C["提取多个中间 Block"]
    C --> D["Token 转二维特征图"]
    D --> E["双线性插值"]
    E --> F["语义特征 P3 / P4 / P5"]
    A --> G["SpatialPriorModulev2"]
    G --> H["卷积下采样"]
    H --> I["细节特征 P3 / P4 / P5"]
    F --> J["通道拼接 Concat"]
    I --> J
    J --> K["1×1 Conv"]
    K --> L["SyncBatchNorm"]
    L --> M["统一通道数的多尺度特征"]
~~~

需要特别区分两个概念：

- 双线性插值本身是无参数操作。
- 整个 STA 并不是完全无参数模块。空间先验分支包含卷积，融合阶段也包含可学习的投影层和归一化层。

### 4.6 S、M、L、X 的 DINOv3 选择

论文给出的 Backbone 设计如下：

| 规格 | Backbone | Hidden Dimension | 说明 |
|---|---|---:|---|
| S | ViT-Tiny | 192 | 由 DINOv3 ViT-Small 蒸馏 |
| M | ViT-Tiny+ | 256 | 由 DINOv3 ViT-Small 蒸馏 |
| L | ViT-Small | 384 | 使用 DINOv3 预训练权重 |
| X | ViT-Small+ | 384 | 使用 DINOv3 预训练权重 |

四种模型均保持 12 层 ViT 深度。

这里需要注意：DEIMv2 论文说明了 S/M Backbone 通过 DINOv3 蒸馏获得，但没有在该论文中完整展开蒸馏训练的所有细节。因此，不应仅凭 DEIMv2 论文推断其蒸馏损失、数据配比和训练时长。

## 五、HGNetv2 轻量化路线

对于 Nano、Pico、Femto 和 Atto，直接使用 DINOv3 ViT 会带来不必要的计算和存储开销。

因此 DEIMv2 选择 HGNetv2-B0 作为轻量化设计基础，通过裁剪网络深度和通道宽度形成不同规格。

### 5.1 不同规格的裁剪方式

论文描述了以下结构变化：

- Nano：使用 HGNetv2 的轻量版本。
- Pico：移除 HGNetv2-B0 的第四阶段，输出保留到 1/16 尺度。
- Femto：在 Pico 的基础上，进一步将最后阶段的 Block 数量从 2 个减少到 1 个。
- Atto：在 Femto 的基础上，进一步将最后一个 Block 的通道数从 512 缩减到 256。

这样可以逐步降低模型参数量和计算量。

### 5.2 为什么轻量模型使用 LiteEncoder？

Pico、Femto、Atto 的 Backbone 输出尺度和通道配置与 DINOv3 路线不同，使用完整的三尺度 HybridEncoder 并不划算。

官方实现为这些模型提供了 LiteEncoder：

- 输入主要为 1/16 特征。
- 通过轻量下采样构造 1/32 特征。
- 使用 GAP Fusion 融合全局信息。
- 采用简化的 FPN/PAN 结构。

以 Pico 配置为例：

- Backbone 输出：512 通道，步长 16。
- Encoder Hidden Dimension：112。
- Encoder 输出：两个尺度，1/16 和 1/32。
- Decoder：两个 Feature Level、3 层 Decoder、200 个 Queries。

相比之下，DEIMv2-S 使用三个尺度、4 层 Decoder 和 300 个 Queries。

这也是为什么不同规格不应该仅根据参数量比较，还需要结合其输入尺寸、特征尺度和 Decoder 配置理解。

## 六、HybridEncoder 与 LiteEncoder

### 6.1 HybridEncoder

DINOv3 路线输出三个尺度：

$$
P_3,\ P_4,\ P_5
$$

对应步长：

$$
[8,16,32]
$$

HybridEncoder 的主要任务是：

1. 将不同来源的特征投影到统一通道维度。
2. 在高层特征上进行 Transformer 特征交互。
3. 通过 FPN/PAN 完成跨尺度信息融合。
4. 输出供 Decoder 使用的多尺度特征。

在 S 模型的配置中：

- 输入通道：[192, 192, 192]。
- Hidden Dimension：192。
- use_encoder_idx 为 [2]，即对指定的高层尺度进行 Transformer 编码。
- num_encoder_layers 为 1。
- fuse_op 为 sum。

### 6.2 LiteEncoder

LiteEncoder 面向极轻量级模型。

其核心流程：

$$
F'=F+\operatorname{GAP}(F)
$$

然后通过卷积和归一化处理。

~~~mermaid
flowchart TD
    A["HGNetv2 输出 1/16 特征"] --> B["1×1 Conv 通道投影"]
    B --> C["AvgPool 下采样"]
    C --> D["1/32 特征"]
    D --> E["GAP Fusion"]
    B --> F["1/16 特征"]
    E --> G["FPN 融合"]
    F --> G
    G --> H["RepNCSPELAN 特征提取"]
    H --> I["PAN 融合"]
    I --> J["输出两个尺度特征"]
~~~

GAP 是 Global Average Pooling，它提供全局上下文信息。

LiteEncoder 不是简单删除一部分 FPN 层，而是围绕两尺度输入重新组织特征融合流程。

## 七、DEIM Transformer Decoder

DEIMv2 的 Decoder 是另一处重要改动。

它基于 Deformable DETR / D-FINE 风格的 Decoder，并引入多个效率优化设计。

### 7.1 Encoder 产生初始候选

Encoder 输出多尺度特征后，会生成初始分类和边界框预测。

随后通过 Top-K 选择得到候选 Queries：

$$
Q=\operatorname{TopK}(\text{Encoder Scores})
$$

这些候选包含：

- Query Content：候选特征内容。
- Reference Boxes：初始参考框。
- Query Position Embedding：由参考框位置生成的位置编码。

DEIMv2 默认使用 300 个 Queries；轻量配置可以减少到 200。

### 7.2 Decoder Layer

每一层主要包含：

1. Self-Attention：不同 Object Queries 之间的信息交互。
2. Multi-Scale Deformable Cross-Attention：从多尺度特征中提取与参考位置相关的信息。
3. SwiGLU FFN：进行特征变换。
4. RMSNorm：归一化。
5. Residual Connection：残差连接。

结构如下：

~~~mermaid
flowchart TD
    A["Query Features"] --> B["Self-Attention"]
    B --> C["Residual + RMSNorm"]
    C --> D["Multi-Scale Deformable Cross-Attention"]
    E["多尺度 Encoder Features"] --> D
    D --> F{"Cross-Attention 融合方式"}
    F --> G["Gateway"]
    F --> H["Residual + RMSNorm"]
    G --> I["SwiGLU FFN"]
    H --> I
    I --> J["Residual + RMSNorm"]
    J --> K["更新后的 Query Features"]
    K --> L["分类与边界框迭代预测"]
~~~

官方代码中提供了 Gateway 机制。开启时使用 Gate 处理 Cross-Attention 输出；关闭时使用残差连接和 RMSNorm。具体采用哪一种由模型配置决定。

### 7.3 SwiGLU FFN

DEIMv2 使用 SwiGLU FFN 替换传统 FFN。

其基本形式为：

$$
\operatorname{SwiGLU}(x)=
\operatorname{SiLU}(xW_g)\odot(xW_u)
$$

再经过输出投影：

$$
\operatorname{FFN}(x)=
\left[
\operatorname{SiLU}(xW_g)
\odot(xW_u)
\right]W_d
$$

其中：

- $W_g$：Gate Projection。
- $W_u$：Up Projection。
- $W_d$：Down Projection。
- $\odot$：逐元素相乘。

它通过门控机制控制特征信息的通过程度。

### 7.4 RMSNorm

RMSNorm 不像 LayerNorm 那样减去均值，而是根据均方根进行缩放：

$$
\operatorname{RMSNorm}(x)=
\frac{x}
{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
\odot\gamma
$$

其中：

- $d$：特征维度。
- $\epsilon$：数值稳定项。
- $\gamma$：可学习缩放参数。

论文使用 RMSNorm 替换常规 LayerNorm，以简化 Decoder 中的归一化计算。

### 7.5 共享 Query Position Embedding

DEIMv2 观察到，在 Decoder 逐层迭代更新过程中，Object Query 的位置变化通常较小。

因此，论文提出在不同 Decoder Layer 之间共享 Query Position Embedding，减少重复计算。

官方代码中先根据初始 Reference Points 计算 Query Position Embedding，再在 Decoder Layer 中复用；如果使用宽层扩展，还会根据特征维度变化进行调整。

### 7.6 D-FINE 风格的边界框分布细化

DEIMv2 的边界框回归保留了 D-FINE 的 Fine-grained Distribution Refinement 思路。

传统回归直接预测连续的边界框坐标，而 D-FINE 将边界距离建模为离散分布，再根据分布计算最终位置。

对于一个边界距离，可以将预测表示为：

$$
p=[p_0,p_1,\ldots,p_R]
$$

通过离散概率分布与非均匀权重函数得到连续距离。概念上可以表示为：

$
d=\sum_{i=0}^{R}p_i\cdot w_i
$

其中 $w_i$ 是由 D-FINE 的非均匀 Weighting Function 生成的投影值，并非简单的等间距整数索引。

其中 $R$ 为最大离散区间数，DEIMv2 配置中常见 reg_max=32。

这种方式能够保留更细粒度的定位信息。

## 八、Dense O2O、MAL 与 Copy-Blend

DEIMv2 的训练改进继承自 DEIM，并在此基础上增加对象级数据增强。

### 8.1 为什么需要 Dense O2O？

传统 DETR 使用 Hungarian Matching 进行 One-to-One 匹配。

假设一张图像中有 $N$ 个真实目标，模型需要从候选预测中找到与这些目标对应的匹配项。

当每张图像中的目标数量较少时，参与正样本监督的目标数量也有限。

Dense O2O 的思路是增加每张训练图像中的有效目标数量，让模型在一次训练中获得更丰富的目标监督。

注意：

**Dense O2O 并不是把 Hungarian Matching 改成传统 YOLO 式的一对多分配。**

它仍然保留 O2O 匹配思路，重点是通过数据增强增加图像中的有效目标数量。

### 8.2 Mosaic 和 MixUp

DEIM 使用图像级数据增强：

- Mosaic：将多张图片组合成一张训练图片。
- MixUp：对两张图片进行加权混合。

这样一张训练图片中就可能包含来自多个原始图片的目标。

### 8.3 Copy-Blend

DEIMv2 进一步提出对象级 Copy-Blend。

其核心思想是：

1. 从 Batch 中收集符合条件的目标。
2. 随机选择若干目标。
3. 根据目标框从源图像中提取图像区域。
4. 将目标区域放到另一张图像中的随机位置。
5. 通过混合方式融合目标区域与目标图像。
6. 同步更新目标框和类别标签。

与普通 Copy-Paste 的区别在于，Copy-Blend 不只是直接覆盖目标区域，而是通过混合方式融合图像内容。

官方实现中可以配置：

- copyblend_prob：启用概率。
- copyblend_epochs：生效训练区间。
- area_threshold：目标面积过滤阈值。
- num_objects：每次选取的目标数量。
- with_expand：是否扩展目标区域。
- expand_ratios：扩展比例。
- copyblend_type：使用 blend 或 copy。

需要注意一个源码实现细节：当前公开代码通过标注框裁剪图像 Patch，而不是依赖独立实例分割 Mask。因此，裁剪 Patch 是否包含目标周围背景，取决于标注框范围。论文中对 Copy-Blend 的方法描述与代码中的具体裁剪实现，应当区分理解。

### 8.4 训练增强流程

~~~mermaid
flowchart TD
    A["原始训练图片和标注"] --> B["基础数据增强"]
    B --> C["Mosaic"]
    C --> D["随机颜色与几何变换"]
    D --> E["MixUp"]
    E --> F["Copy-Blend"]
    F --> G["更新图片与目标标注"]
    G --> H["Backbone + Encoder"]
    H --> I["Decoder 预测"]
    I --> J["Hungarian Matching"]
    J --> K["计算 MAL / BBox / GIoU"]
    J --> L["计算 FGL / DDF"]
    K --> M["加权总损失"]
    L --> M
    M --> N["反向传播和参数更新"]
~~~

不同模型规格的增强策略并不完全相同。特别是极轻量级模型，不能机械套用 S、M、L、X 的全部增强和损失配置。

## 九、损失函数

DEIMv2 的总损失由五部分组成：

1. Matchability-Aware Loss（MAL）
2. Fine-Grained Localization Loss（FGL）
3. Decoupled Distillation Focal Loss（DDF）
4. L1 Bounding Box Loss
5. GIoU Loss

整体可以表示为：

$$
\begin{aligned}
L_{\text{total}}={}&
\lambda_1L_{\text{MAL}}
+\lambda_2L_{\text{FGL}}
+\lambda_3L_{\text{DDF}}\\
&+\lambda_4L_{\text{bbox}}
+\lambda_5L_{\text{GIoU}}
\end{aligned}
$$

论文中使用的权重为：

$$
\begin{aligned}
\lambda_1&=1.0\\
\lambda_2&=0.15\\
\lambda_3&=1.5\\
\lambda_4&=5.0\\
\lambda_5&=2.0
\end{aligned}
$$

### 9.1 MAL

MAL 是 DEIM 提出的 Matchability-Aware Loss。

它不仅关注预测类别是否正确，还将匹配质量纳入分类监督。

对于匹配到的预测，使用预测框与真实框的 IoU 作为质量信息，形成质量感知的分类目标。

官方实现中，匹配目标的 IoU 会参与目标分数计算；同时利用预测置信度构造动态权重。

可以将其核心思想理解为：

- 匹配质量越高，分类目标分数越高。
- 低质量匹配不会与高质量匹配得到完全相同的监督。
- 动态权重使分类损失能够感知预测与真实目标之间的匹配质量。

### 9.2 FGL

FGL 全称 Fine-Grained Localization Loss。

它来自 D-FINE 的细粒度定位思想，主要用于提升边界框定位精度。

它将边界距离离散化为分布，并根据目标位置计算分布监督，使模型学习更细致的边界位置。

### 9.3 DDF

DDF 全称 Decoupled Distillation Focal Loss。

它同样来自 D-FINE 的局部定位学习机制，通过分布层面的监督进一步改善边界框回归。

FGL 和 DDF 主要用于 DINOv3 路线以及 Nano 等具有足够模型容量的配置。

### 9.4 BBox Loss 与 GIoU Loss

L1 边界框损失：

$$
L_{\text{bbox}}=
\sum_i
\left\|
b_i-\hat b_i
\right\|_1
$$

GIoU Loss：

$$
L_{\text{GIoU}}=1-\operatorname{GIoU}(b,\hat b)
$$

L1 关注坐标差异，GIoU 则从几何重叠关系上衡量预测框与真实框之间的差异。

### 9.5 不同模型使用的损失不同

论文指出，FGL 和 DDF 对部分超轻量模型并不一定有利。

当模型容量较小、基础预测能力不足时，使用局部分布损失可能无法获得预期收益。

因此：

- X、L、M、S：使用 MAL、FGL、DDF、BBox、GIoU。
- Nano：使用完整损失配置。
- Pico、Femto、Atto：不使用 FGL 和 DDF，保留 MAL、BBox、GIoU。

官方 Pico 配置中可以看到：

~~~yaml
DEIMCriterion:
  losses: ['mal', 'boxes']
~~~

这意味着极轻量模型不应该直接照搬 S 模型的全部损失配置。

## 十、模型规格与 COCO 实验结果

### 10.1 八种模型对比

下表采用官方 GitHub 仓库当前 Model Zoo 的数据。参数量和 GFLOPs 为官方报告值，数值经过四舍五入。

| 模型 | 输入尺寸 | COCO AP | 参数量 | GFLOPs |
|---|---:|---:|---:|---:|
| Atto | 320 | 23.8 | 0.5M | 0.8 |
| Femto | 416 | 31.0 | 1.0M | 1.7 |
| Pico | 640 | 38.5 | 1.5M | 5.2 |
| Nano（N） | 640 | 43.0 | 3.6M | 6.8 |
| S | 640 | 50.9 | 9.7M | 49.5 |
| M | 640 | 53.0 | 18.1M | 84.1 |
| L | 640 | 56.0 | 32.2M | 144.5 |
| X | 640 | 57.8 | 50.3M | 200.0 |

### 10.2 一个值得注意的 FLOPs 数据变化

论文早期表格中的部分 GFLOPs 数值与官方仓库当前 Model Zoo 不一致。

例如：

| 模型 | 论文早期表格 | 官方仓库修正后 |
|---|---:|---:|
| S | 25.62 | 49.5 |
| M | 52.20 | 84.1 |
| L | 96.32 | 144.5 |
| X | 151.6 | 200.0 |

官方仓库在 2026 年 10 月 22 日的更新说明中指出，之前的 FLOPs 计算逻辑遗漏了 Attention FLOPs，之后修复了这一问题。

因此，阅读论文或引用实验数据时，需要留意使用的是哪个版本的计算逻辑。本文优先使用官方仓库修正后的数值。

### 10.3 主要实验结论

论文中有几个值得关注的结果：

**DEIMv2-S：**

- 参数量约 9.71M。
- COCO AP 达到 50.9。
- 是论文强调的首个参数量低于 10M、AP 超过 50 的模型。

**DEIMv2-X：**

- 参数量约 50.3M。
- COCO AP 达到 57.8。
- 相比 DEIM-X 的 56.5 AP，参数量和计算量都有所降低。

**DEIMv2-Pico：**

- 参数量约 1.5M。
- COCO AP 达到 38.5。
- 与 YOLOv10-Nano 的 38.5 AP 相当，但参数量更少。

这些结果说明 DEIMv2 的价值不仅体现在最大模型的精度上，还在于它覆盖了从极小模型到高性能模型的完整规格范围。

### 10.4 如何理解这些结果？

需要注意以下几点：

1. COCO AP 是特定数据集和评估协议下的结果，不代表所有数据集都能达到相同表现。
2. 不同模型的输入尺寸不同，不能只根据参数量判断实际速度。
3. GFLOPs 不等于实际推理延迟，实际速度还受算子实现、显存访问、硬件和推理引擎影响。
4. DINOv3 路线和 HGNetv2 路线在特征结构上有差异，不能仅凭同一参数量进行横向推断。
5. 论文中的训练 Epoch 数量、官方配置中的总 Epoch 数量可能因为 EMA 延长训练等设置而不同。

## 十一、训练配置与自定义数据集

### 11.1 环境准备

官方仓库建议使用 Python 3.11，并给出 PyTorch 2.4.1 / 2.5.1 的测试参考。

~~~bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git
cd DEIMv2

conda create -n deimv2 python=3.11 -y
conda activate deimv2

pip install -r requirements.txt
~~~

具体环境应以仓库当前 requirements.txt 和运行时依赖为准。

### 11.2 Backbone 权重准备

不同路线的预训练权重不同：

| 模型规格 | 需要准备的 Backbone |
|---|---|
| S | DINOv3 蒸馏得到的 ViT-Tiny |
| M | DINOv3 蒸馏得到的 ViT-Tiny+ |
| L | DINOv3 ViT-Small |
| X | DINOv3 ViT-Small+ |
| Nano / Pico / Femto / Atto | HGNetv2 对应权重，按官方配置自动下载或加载 |

官方仓库要求将相关 ViT 权重放入 ckpts/，例如：

~~~text
ckpts/
├── dinov3_vits16.pth
├── vitt_distill.pt
└── vittplus_distill.pt
~~~

不要把 ViT-Tiny、ViT-Tiny+、DINOv3-S 和 DINOv3-S+ 的权重混用。权重必须与模型配置匹配。

### 11.3 自定义数据集格式

官方仓库支持 COCO 格式数据集。

~~~text
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
~~~

自定义类别时，需要关闭 COCO 类别映射：

~~~yaml
remap_mscoco_category: False
~~~

同时修改 num_classes 为你的类别数量。

数据加载配置需要分别指定训练和验证图像目录、标注文件以及 COCO Evaluator。

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

测试：

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

以上命令中的配置文件需要与所使用的模型规格对应。

### 11.5 S 模型配置示例

官方 S 配置中的关键部分：

~~~yaml
DEIM:
  backbone: DINOv3STAs

DINOv3STAs:
  name: vit_tiny
  embed_dim: 192
  weights_path: ./ckpts/vitt_distill.pt
  interaction_indexes: [3, 7, 11]
  num_heads: 3

HybridEncoder:
  in_channels: [192, 192, 192]
  depth_mult: 0.67
  expansion: 0.34
  hidden_dim: 192
  dim_feedforward: 512

DEIMTransformer:
  feat_channels: [192, 192, 192]
  hidden_dim: 192
  dim_feedforward: 512
  num_layers: 4
  eval_idx: -1
~~~

这里的 interaction_indexes 是源码配置中的索引值。不同权重版本的配置可能不同，使用时应直接采用对应 checkpoint 发布时的配置。

### 11.6 训练 Epoch 与 EMA

DEIMv2 的训练配置中存在额外的 EMA 训练安排。

例如官方 S 配置：

~~~yaml
epoches: 132
~~~

其注释说明配置总 Epoch 数量包含用于 EMA 的额外训练安排。论文表格中的训练 Epoch 与代码配置中的总 Epoch 数量并不一定直接相等。

修改训练轮数时，还需要同步调整：

- flat_epoch
- no_aug_epoch
- Mosaic / MixUp 的生效区间
- Copy-Blend 的生效区间
- matcher_change_epoch
- EMA 相关配置

不要只修改 epoches 一个参数。

## 十二、Hugging Face 模型使用

DEIMv2 官方模型已经上传至 Hugging Face。

部分模型入口：

| 模型 | Hugging Face |
|---|---|
| DINOv3-S | [DEIMv2_DINOv3_S_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO) |
| DINOv3-M | [DEIMv2_DINOv3_M_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO) |
| DINOv3-L | [DEIMv2_DINOv3_L_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO) |
| DINOv3-X | [DEIMv2_DINOv3_X_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO) |
| HGNetv2-Pico | [DEIMv2_HGNetv2_PICO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO) |

官方仓库使用 PyTorchModelHubMixin 加载模型。需要先准备官方 DEIMv2 仓库中的 engine 代码和对应模型配置。

简化后的使用结构如下：

~~~python
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from engine.backbone import DINOv3STAs
from engine.deim import HybridEncoder, DEIMTransformer
from engine.deim.postprocessor import PostProcessor


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        self.encoder = HybridEncoder(**config["HybridEncoder"])
        self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        self.postprocessor = PostProcessor(**config["PostProcessor"])

    def forward(self, x, orig_target_sizes):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)
        return x
~~~

加载权重：

~~~python
model = DEIMv2.from_pretrained(
    "Intellindust/DEIMv2_DINOv3_S_COCO"
)
~~~

以上是官方仓库 README 中的模型封装思路。实际使用时还必须提供与模型权重对应的完整配置，不能只实例化模型而不加载配置。

### 12.1 Transformers 集成

Hugging Face Transformers 文档也提供了 DEIMv2 对应的配置和模型类：

- Deimv2Config
- Deimv2ForObjectDetection

这是一条与官方仓库 PyTorchModelHubMixin 使用方式不同的集成路径。具体模型仓库、权重格式和 Transformers 版本需要按照对应模型卡与文档确认，不能默认所有官方 safetensors 权重都能直接通过任意版本的 AutoModelForObjectDetection 加载。

## 十三、ONNX 与 TensorRT 部署

官方仓库提供 ONNX 导出脚本：

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

TensorRT 转换：

~~~bash
trtexec \
--onnx="model.onnx" \
--saveEngine="model.engine" \
--fp16
~~~

官方仓库特别提示：

- 推荐 TensorRT 10.6 或更高版本进行 FP16 推理。
- TensorRT 10.4 的 FP16 推理存在可能产生错误输出的问题。
- 如果使用旧版本，需要考虑 FP32 推理，并对整个推理流程进行数值和检测精度验证。

### 13.1 导出输入与输出

官方 ONNX 导出脚本中，模型输入包括：

- images
- orig_target_sizes

输出包括：

- labels
- boxes
- scores

导出脚本将模型和 PostProcessor 封装在一起，最终输出经过后处理的检测结果。

这意味着部署时不能只关注 Backbone 和 Decoder，还需要保证图像预处理、原始尺寸输入、坐标缩放和输出解释与 PyTorch 版本一致。

### 13.2 后处理是否需要 NMS？

DEIMv2 使用 DETR 风格的集合预测和匹配训练，不依赖传统 YOLO 检测流程中的标准 NMS 作为核心后处理。

官方 PostProcessor 在 Focal Loss 模式下：

1. 对分类 Logits 使用 Sigmoid。
2. 展平 Query 与类别维度。
3. 选择 Top-K 分数。
4. 根据索引恢复类别和 Query。
5. 取出对应边界框。
6. 根据原始图像尺寸恢复坐标。

因此，部署时需要重点检查 Top-K 输出、类别索引和边界框坐标转换，而不是直接套用 YOLO 的 NMS 后处理。

## 十四、优点、局限性与适用场景

### 14.1 优点

1. **DINOv3 特征迁移**：通过 STA 将基础模型语义特征适配到检测任务。
2. **多尺度特征构建**：使用中间层语义与空间先验分支，兼顾语义和细节。
3. **模型规格丰富**：从 Atto 到 X，覆盖不同硬件预算。
4. **训练监督更充分**：Dense O2O 与 Copy-Blend 增加有效目标监督。
5. **Decoder 更轻量**：SwiGLU、RMSNorm 和共享 Query Position Embedding 减少不必要的计算。
6. **部署工具相对完整**：官方提供 ONNX、TensorRT 和推理脚本。

### 14.2 局限性与需要注意的地方

1. DINOv3 路线依赖对应的预训练或蒸馏 Backbone 权重，权重和配置必须匹配。
2. DINOv3 ViT 路线的计算量并不低，不能仅根据参数量判断端侧部署成本。
3. STA 增加了特征适配分支，实际速度需要通过目标硬件测试。
4. 极轻量 HGNetv2 路线与 DINOv3 路线的结构不同，性能和部署特性不能简单等同。
5. Copy-Blend 的数据增强效果依赖目标标注质量、目标区域裁剪和混合策略。
6. 论文中的 FLOPs 与官方仓库后续修正数据存在差异，引用时必须标明数据来源。
7. COCO 上的 AP 不代表在密集农业目标、小尺寸植物和复杂遮挡数据集上一定有同样表现。

### 14.3 适用场景

| 场景 | 可考虑的规格 |
|---|---|
| 移动设备、极低算力 | Atto / Femto |
| 轻量边缘设备 | Pico / Nano |
| 中等算力 GPU | S / M |
| 高精度 GPU 检测 | L / X |

最终选择仍应以目标数据集上的 AP、召回率、小目标指标、推理延迟、显存占用和部署兼容性为准。

## 十五、总结

DEIMv2 的核心并不是某一个全新的 Attention 或卷积模块，而是将多项设计组合成一套完整的实时检测体系：

- 用 DINOv3 提供高质量语义特征。
- 用 STA 将单尺度 ViT 特征转为检测需要的多尺度特征。
- 用 HGNetv2 和裁剪策略覆盖极轻量级模型。
- 用 HybridEncoder / LiteEncoder 适配不同 Backbone。
- 用简化后的 DEIM Transformer Decoder 完成端到端检测。
- 用 Dense O2O、MAL 和 Copy-Blend 改善训练监督。
- 用 FGL、DDF、L1 和 GIoU 完成分类与定位优化。

从技术路线看，DEIMv2 是一项将视觉基础模型与实时 DETR 检测器结合的系统性工程工作。它的价值在于提供了从高精度到超轻量的完整模型系列，而不只是追求单一规格上的最高 AP。

对于实际项目，建议优先从官方 S、M 或轻量 HGNetv2 模型开始，在自己的数据集上完成微调和部署验证，再决定是否需要更大的模型。

## 参考资料

### 论文

1. [Real-Time Object Detection Meets DINOv3（DEIMv2）](https://arxiv.org/abs/2509.20787)
2. [DEIM: DETR with Improved Matching for Fast Convergence（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html)
3. [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)

### 官方代码与配置

4. [DEIMv2 官方 GitHub 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2)
5. [DEIMv2 S 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_s_coco.yml)
6. [DEIMv2 Pico 配置](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_hgnetv2_pico_coco.yml)
7. [DINOv3 STA Backbone 实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/dinov3_adapter.py)
8. [DEIM Decoder 实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py)
9. [DEIM Loss 实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_criterion.py)
10. [Copy-Blend 数据增强实现](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/data/dataloader.py)
11. [ONNX 导出脚本](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/tools/deployment/export_onnx.py)

### Hugging Face

12. [Intellindust 官方模型组织](https://huggingface.co/Intellindust)
13. [DEIMv2 DINOv3-S 权重](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO)
14. [DEIMv2 HGNetv2-Pico 权重](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO)
15. [Hugging Face Transformers DEIMv2 文档](https://huggingface.co/docs/transformers/main/model_doc/deimv2)
