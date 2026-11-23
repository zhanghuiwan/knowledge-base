---
title: YOLOv1：网络结构、Confidence、Loss 与完整训练推理解析
description: 按照四阶段视觉模型研究框架，结合 YOLOv1 原论文与 pjreddie/darknet 源码，深入解析网格预测、30 维输出、Confidence、Responsible Predictor、完整 Loss 数值计算、网络数据流、训练策略、NMS 与模型局限。
---

# YOLOv1：网络结构、Confidence、Loss 与完整训练推理解析

## 阅读说明

本文以 Joseph Redmon 等人在 CVPR 2016 发表的论文 **You Only Look Once: Unified, Real-Time Object Detection** 为主要研究对象，并结合作者的 [Darknet 仓库](https://github.com/pjreddie/darknet)检查历史配置和检测层实现。

文章按照四阶段视觉模型研究框架组织：

- **阶段 A：理解模型**——研究背景、任务定义、输出表示、网络结构和单张图片的前向传播。
- **阶段 B：理解训练与推理**——研究初始化、正负样本分配、Confidence、完整 Loss、反向传播、解码和 NMS。
- **阶段 C：评价模型**——研究实验数据、精度与速度、论文可视化、改进价值和局限性。
- **阶段 D：复现与应用**——将论文设计对应到 Darknet 配置和源码，说明复现时容易混淆的版本差异。

**资料边界：**

1. 论文中的主要 YOLO 模型使用 $S=7$、$B=2$、$C=20$，输出为 $7\times7\times30$。
2. 当前 Darknet 仓库中的 `cfg/yolov1.cfg` 使用 `num=3`，输出为 1715 个数，并包含论文结构描述中没有直接对应的实现细节。不能把该配置当作论文实验设置的逐字复刻。
3. 文章中的数值例子用于解释论文公式；涉及 Darknet 特定行为时会明确标记为源码实现。
4. YOLOv1 的最终输出层使用线性激活。论文将输出解释为类别概率和 Confidence，但它们不是由 Softmax / Sigmoid 强制限制在 $[0,1]$ 内的概率分布。

论文：[CVPR 2016 原文 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)。

---

# 阶段 A：理解模型（Understand the Model）

## 1. 研究背景与问题定义

### 1.1 YOLOv1 要解决什么问题？

YOLOv1 出现之前，主流目标检测方法通常采用多阶段流程。以 R-CNN 系列为例，典型步骤包括：

1. 生成候选区域（Region Proposal）。
2. 对候选区域提取特征。
3. 对每个区域进行分类。
4. 回归边界框。
5. 通过 NMS 等后处理去除重复检测。

这类方法将候选区域生成、分类、定位等任务拆分成多个环节，计算流程较复杂，也不容易作为一个统一模型端到端优化。

YOLOv1 的核心想法是：

> 将目标检测直接建模为从整张图片到目标位置和类别的回归问题，让一个神经网络在一次前向传播中同时预测所有目标。

输入一张完整图片，网络直接输出：

- 目标边界框的位置和尺寸。
- 每个预测框包含目标的 Confidence。
- 每个网格中的条件类别概率。

整体流程：

~~~mermaid
flowchart LR
    A["完整输入图片"] --> B["单个 CNN"]
    B --> C["网格级预测"]
    C --> D["Bounding Boxes + Confidence + Class Scores"]
    D --> E["置信度阈值过滤"]
    E --> F["NMS"]
    F --> G["最终检测结果"]
~~~

这并不意味着 YOLOv1 完全没有后处理。它取消了传统的候选区域生成和逐区域分类流程，但仍然使用置信度过滤和 NMS。

### 1.2 YOLOv1 的核心设计

| 设计 | 具体做法 | 目的 |
|---|---|---|
| 整图预测 | 网络一次处理完整图片 | 利用全局上下文，避免逐区域运行分类器 |
| Grid 划分 | 将图片划分为 $S\times S$ 个网格 | 将目标位置预测分配到空间区域 |
| 直接框回归 | 每个网格预测多个 Bounding Box | 直接预测位置和尺寸 |
| Confidence | 每个预测框预测一个 Confidence | 同时表达目标存在性与定位质量 |
| 网格级分类 | 每个网格预测一组条件类别概率 | 共享类别预测，减少输出维度 |
| 统一 Loss | 坐标、Confidence、分类误差共同优化 | 端到端训练 |

YOLOv1 最重要的思想不是某个复杂模块，而是把整张图片的检测任务统一成一个密集的回归问题。

### 1.3 论文中需要记住的模型配置

PASCAL VOC 检测任务使用：

| 参数 | 数值 | 含义 |
|---|---:|---|
| 输入尺寸 | $448\times448$ | 图片宽度和高度 |
| $S$ | 7 | 每个方向的网格数 |
| $S^2$ | 49 | 网格总数 |
| $B$ | 2 | 每个网格预测的 Bounding Box 数量 |
| $C$ | 20 | VOC 类别数量 |
| 每个网格输出 | 30 | $2\times5+20$ |
| 最终输出 | $7\times7\times30$ | 全部网格的预测结果 |

注意：本文使用 $N$ 表示 Batch Size，使用 $B$ 表示论文中的每网格预测框数量，避免两个概念都写成 B。

---

## 2. 任务定义与输入输出表示（Task Formulation）

### 2.1 输入图片

以 Batch Size 为 1、输入尺寸为 $448\times448$ 为例。

原始图片可以表示为：

$$
X\in\mathbb{R}^{N\times H\times W\times3}
$$

其中：

- $N=1$：Batch Size。
- $H=W=448$：图片尺寸。
- 3：RGB 通道。

在常见 PyTorch CNN 实现中，通常转换为 NCHW：

$$
X\in\mathbb{R}^{1\times3\times448\times448}
$$

论文中的图片输入为 $448\times448$，并不意味着网络只处理某个局部裁剪区域。网络始终接收整张图片。

### 2.2 Grid：为什么要划分网格？

YOLOv1 将图片划分为：

$$
S\times S=7\times7
$$

如果某个目标的中心点落在网格 $i$ 内，该网格就负责预测这个目标。

例如目标中心点相对于整张图片的归一化坐标为：

$$
(x_c,y_c)=(0.5,0.5)
$$

它所在的网格索引为：

$$
g_x=\lfloor 0.5\times7\rfloor=3
$$

$$
g_y=\lfloor 0.5\times7\rfloor=3
$$

采用从 0 开始的索引时，它由第 4 行、第 4 列的网格负责。

~~~mermaid
flowchart TD
    A["输入图片"] --> B["划分为 7×7 Grid"]
    C["GT Bounding Box"] --> D["计算 GT 中心点"]
    B --> E["找到中心点所在 Grid"]
    D --> E
    E --> F["该 Grid 负责预测该目标"]
~~~

为什么使用中心点分配？

一个目标的边界框可能覆盖多个网格。如果每个覆盖区域都负责检测，就容易产生大量重复预测。YOLOv1 使用中心点规则，让目标主要由一个网格负责。

这种设计简单，但也带来空间约束：如果多个目标的中心点落在同一个网格，网格级类别预测就会成为瓶颈。

### 2.3 一个网格预测什么？

每个网格预测：

- $B=2$ 个 Bounding Box。
- 每个 Bounding Box 包含 5 个数值。
- 一组 $C=20$ 维条件类别概率。

每个 Bounding Box：

$$
(x,y,w,h,Confidence)
$$

各字段含义：

| 字段 | 含义 | 坐标参考 |
|---|---|---|
| $x$ | Bounding Box 中心点横坐标 | 相对于当前网格 |
| $y$ | Bounding Box 中心点纵坐标 | 相对于当前网格 |
| $w$ | Bounding Box 宽度 | 相对于整张图片 |
| $h$ | Bounding Box 高度 | 相对于整张图片 |
| Confidence | 目标存在性与定位质量的综合分数 | 每个预测框独立 |

其中 $x,y,w,h$ 都使用归一化坐标。训练目标中的 $x,y$ 被表示为网格内部的偏移量，$w,h$ 则相对于整张图片的宽高归一化。

### 2.4 30 个输出值的组成

VOC 数据集有 20 个类别，每个网格输出：

$$
2\times5+20=30
$$

可以从逻辑上将一个网格的输出组织为：

$$
Y_i=[Box_1,Box_2,ClassProbabilities]
$$

展开：

$$
\begin{aligned}
Y_i=[&
x_1,y_1,w_1,h_1,C_1,\\
&x_2,y_2,w_2,h_2,C_2,\\
&p_1,p_2,\ldots,p_{20}]
\end{aligned}
$$

所以：

$$
Y\in\mathbb{R}^{N\times7\times7\times30}
$$

注意，这是一种方便理解的逻辑排列。Darknet 的历史 `detection_layer` 实现会按类别、Confidence 和坐标分段存放扁平数组，不能默认其原始内存排列与这里的教学排列完全相同。

### 2.5 类别概率为什么是网格级的？

每个网格只预测一组条件类别概率：

$$
P(Class_c\mid Object)
$$

它表示：

> 已知当前网格包含目标，该目标属于类别 c 的条件概率。

YOLOv1 并不是每个 Bounding Box 都有独立的 20 维类别向量。

两个 Box Predictor 共用同一组类别概率。这种设计节省了输出维度，但也限制了同一个网格同时表达多个不同类别目标的能力。

例如一个网格的类别输出：

$$
P=[0.7,0.2,0.1,0,\ldots,0]
$$

可以理解为：

- 类别 0：0.7。
- 类别 1：0.2。
- 类别 2：0.1。
- 其余类别：0。

这里的类别编号从 0 开始。

### 2.6 输出维度如何计算？

每个网格：

$$
B\times5+C=2\times5+20=30
$$

全部网格：

$$
7\times7\times30=1470
$$

因此网络最后一个全连接层输出 1470 个数，随后可以重新组织为：

$$
[N,1470]\rightarrow[N,7,7,30]
$$

这 1470 个数并不是 1470 个独立目标，而是 49 个网格的预测结果。

---

## 3. Confidence：YOLOv1 最容易混淆的概念

这是理解 YOLOv1 的关键。

### 3.1 论文中的定义

YOLOv1 将每个 Bounding Box 的 Confidence 定义为：

$$
\boxed{Confidence=P(Object)\times IoU(B_{pred},B_{gt})}
$$

其中：

- $P(Object)$：该预测框包含真实目标的概率。
- $IoU(B_{pred},B_{gt})$：预测框与 Ground Truth 的交并比。

IoU 定义：

$$
IoU=\frac{Area(B_{pred}\cap B_{gt})}
{Area(B_{pred}\cup B_{gt})}
$$

这个定义希望 Confidence 同时表达两件事：

1. 预测框是否包含目标。
2. 预测框的位置是否准确。

仅预测目标存在概率是不够的。例如两个框都认为有目标，但一个框定位准确，另一个框偏差很大。如果只看目标存在概率，两者可能得到相近的分数。

把 IoU 纳入 Confidence 后，模型还需要学习预测框的定位质量。

### 3.2 训练时为什么负责预测器的 Confidence 目标等于 IoU？

假设某个网格包含一个 GT。

训练时，模型已经知道这个网格有目标，并且会选择一个预测框负责这个 GT。对于这个负责预测框，目标存在这一条件已经成立，因此训练监督目标为：

$$
P(Object)=1
$$

于是：

$$
Confidence^*=1\times IoU(B_{pred},B_{gt})
$$

即：

$$
\boxed{Confidence^*=IoU(B_{pred},B_{gt})}
$$

例如：

- 当前预测框与 GT 的 IoU 为 0.8。
- 网络输出的 Confidence 为 0.7。

那么 Confidence Loss 为：

$$
L_{conf}=(0.8-0.7)^2=0.01
$$

网络通过反向传播，让 Confidence 预测逐渐接近 0.8。

**重要：网络并没有分别输出 $P(Object)$ 和 IoU 两个数，再在网络外相乘。每个预测框只输出一个 Confidence。** $P(Object)\times IoU$ 是论文对这个输出的语义定义；训练时通过 IoU 构造负责预测框的监督目标。

### 3.3 负责框、非负责框和背景框的 Confidence 目标

对于一个包含 GT 的网格：

- 与 GT 的 IoU 最大的预测框成为 Responsible Predictor。
- Responsible Predictor 的 Confidence 目标为当前预测框与 GT 的 IoU。
- 其他预测框不负责该 GT，在 Darknet 参考实现中受到 No-Object Confidence 惩罚，目标为 0。

对于没有 GT 的网格：

- 所有预测框的 Confidence 目标均为 0。

这里必须区分：

- 网格是否包含 GT。
- 某个预测框是否负责 GT。
- 某个预测框是否受到 No-Object Loss 惩罚。

这三个概念不是同一个指标。后文会结合论文公式和 Darknet 源码解释。

### 3.4 推理时如何使用 Confidence？

推理时没有 GT，无法计算真实 IoU。

网络直接输出预测 Confidence，再乘以网格级的条件类别概率：

$$
\boxed{
Score(c)=\hat C_{ij}\times\hat p_i(c)
}
$$

例如：

- Box Confidence：0.70。
- Cat 类别概率：0.70。
- Dog 类别概率：0.20。
- Person 类别概率：0.10。

则：

| 类别 | 条件类别概率 | Confidence | 最终类别分数 |
|---|---:|---:|---:|
| Cat | 0.70 | 0.70 | 0.49 |
| Dog | 0.20 | 0.70 | 0.14 |
| Person | 0.10 | 0.70 | 0.07 |

这里不需要再乘一次 $P(Object)$，因为 Confidence 已经包含了目标存在性和定位质量的语义。

如果网络学得理想，最终分数可以理解为：

$$
P(Class_c)\times IoU
$$

但实际推理时使用的是网络估计的 Confidence 和类别分数，并不是真实 IoU，也不是严格经过概率校准的统计学概率。

### 3.5 与现代 YOLO 的 Confidence 区别

不要把 YOLOv1 的 Confidence 和现代 Ultralytics YOLO 的 `conf` 混为一谈。

| 模型 | 检测头是否有独立 Objectness | 推理检测分数的主要含义 |
|---|---|---|
| YOLOv1 | 有每框 Confidence，但其目标包含 IoU | Confidence × 条件类别概率 |
| YOLOv5 | 有独立 Objectness 分支 | Objectness × Class Score |
| YOLOv8 / YOLO11 | 无独立 Objectness 分支 | 类别分支分数 |
| YOLO26 | 无独立 Objectness 分支 | 类别分支分数 |

以 YOLO11 为例，分类分数经过 Sigmoid 后直接参与后处理。训练时 Task-Aligned Assigner 会结合分类分数和预测框 IoU 生成对齐的软监督目标，因此分类分支会间接学习定位质量；但推理时不会再显式乘一次 IoU。

这说明不同代际的 `conf` 并不是始终具有完全相同的定义。

---

## 4. 完整网络架构（Architecture Deep Dive）

### 4.1 整体结构

YOLOv1 使用一个 CNN 完成特征提取和检测预测。

论文中的主要网络：

- 24 个卷积层。
- 2 个全连接层。
- 交替使用 1×1 和 3×3 卷积。
- 通过 MaxPool 和部分步长卷积逐步降低空间分辨率。
- 最后通过全连接层输出 1470 个预测值。

~~~mermaid
flowchart TD
    A["输入 448×448×3"] --> B["24 个卷积层"]
    B --> C["特征图 7×7×1024"]
    C --> D["Flatten"]
    D --> E["FC：4096"]
    E --> F["Dropout"]
    F --> G["FC：1470"]
    G --> H["Reshape：7×7×30"]
    H --> I["Bounding Box + Confidence + Class"]
~~~

论文网络结构图见原论文 Figure 3。

### 4.2 为什么使用 1×1 和 3×3 卷积交替？

YOLOv1 的卷积结构受到 GoogLeNet 等网络的启发，但没有直接采用 Inception 模块。

它主要使用：

- **1×1 Conv**：调整通道数、压缩特征维度，减少后续卷积的计算量。
- **3×3 Conv**：提取空间邻域特征，融合局部信息。

例如：

$$
[B,512,H,W]\xrightarrow{1\times1}
[B,256,H,W]
$$

然后：

$$
[B,256,H,W]\xrightarrow{3\times3}
[B,512,H,W]
$$

先压缩通道，再使用 3×3 卷积，可以降低计算成本。

### 4.3 逐阶段 Tensor Shape

以 Batch Size=1、输入 448×448 为例。

| 阶段 | 主要操作 | 输出 Shape（NCHW） |
|---|---|---|
| Input | RGB 图片 | $[1,3,448,448]$ |
| 1 | Conv 7×7，64 通道，Stride 2 | $[1,64,224,224]$ |
| 2 | MaxPool 2×2，Stride 2 | $[1,64,112,112]$ |
| 3 | Conv 3×3，192 通道 | $[1,192,112,112]$ |
| 4 | MaxPool 2×2，Stride 2 | $[1,192,56,56]$ |
| 5 | 多组 1×1 / 3×3 Conv | $[1,512,56,56]$ |
| 6 | MaxPool 2×2，Stride 2 | $[1,512,28,28]$ |
| 7 | 多组 1×1 / 3×3 Conv | $[1,512,28,28]$ |
| 8 | MaxPool 2×2，Stride 2 | $[1,512,14,14]$ |
| 9 | 深层 1×1 / 3×3 Conv | $[1,1024,14,14]$ |
| 10 | Conv 下采样及深层卷积 | $[1,1024,7,7]$ |
| 11 | Flatten | $[1,50176]$ |
| 12 | FC | $[1,4096]$ |
| 13 | FC | $[1,1470]$ |
| 14 | Reshape | $[1,7,7,30]$ |

表格按照论文 Figure 3 的阶段性空间尺寸归纳；具体卷积层重复次数和通道配置应以 Figure 3 为准，不应将后续 Darknet 配置中的额外层直接拼进论文结构。

### 4.4 全连接层的作用与参数规模

最终卷积特征：

$$
F\in\mathbb{R}^{1\times1024\times7\times7}
$$

Flatten 后：

$$
1024\times7\times7=50176
$$

得到：

$$
F_{flat}\in\mathbb{R}^{1\times50176}
$$

第一层全连接：

$$
50176\rightarrow4096
$$

第二层全连接：

$$
4096\rightarrow1470
$$

仅第一层全连接的权重数量就是：

$$
50176\times4096=205,520,896
$$

第二层全连接的权重数量：

$$
4096\times1470=6,021,120
$$

这说明 YOLOv1 的全连接预测头具有很大的参数规模。全连接层也使输出空间尺寸与网络结构绑定，不能像纯卷积检测头那样自然地适配任意输入尺寸。

### 4.5 激活函数与输出层

论文使用 Leaky ReLU：

$$
\phi(x)=
\begin{cases}
x,&x>0\\
0.1x,&x\leq0
\end{cases}
$$

最后一层使用线性激活（Linear Activation）。

这意味着：

- 最终的 $x,y,w,h$ 不经过 Sigmoid。
- Confidence 不经过 Sigmoid。
- 类别输出不经过 Softmax。

论文使用 SSE 将这些输出回归到相应的训练目标。虽然作者将类别输出称为条件类别概率，但线性激活本身不会保证它们处于 $[0,1]$，也不会保证 20 个类别分数之和严格等于 1。

因此，理解 YOLOv1 时应区分**输出的语义定义**和**最后一层实际使用的激活函数**。

### 4.6 网络设计的主要代价

1. 最后的全连接层参数量较大。
2. 输出网格固定为 7×7。
3. 多次下采样使最终检测特征较粗。
4. 每个网格只有两个预测框和一组类别概率。
5. 对小目标、密集目标和特殊宽高比目标的表达能力受限。

---

## 5. 单张图片的完整前向传播（Forward Pass）

现在将一张图片从输入到检测输出完整追踪一遍。

### 5.1 输入

$$
X\in\mathbb{R}^{1\times448\times448\times3}
$$

转换为 NCHW：

$$
X\in\mathbb{R}^{1\times3\times448\times448}
$$

### 5.2 Backbone 特征提取

图片经过 24 个卷积层，空间尺寸逐步降低：

$$
448\rightarrow224\rightarrow112\rightarrow56
\rightarrow28\rightarrow14\rightarrow7
$$

最终得到：

$$
F\in\mathbb{R}^{1\times1024\times7\times7}
$$

这里的特征图每个空间位置对应一个网格预测位置。

### 5.3 全连接预测

Flatten：

$$
[1,1024,7,7]\rightarrow[1,50176]
$$

FC：

$$
[1,50176]\rightarrow[1,4096]
$$

FC：

$$
[1,4096]\rightarrow[1,1470]
$$

Reshape：

$$
[1,1470]\rightarrow[1,7,7,30]
$$

### 5.4 最终 1470 个数的含义

$$
1470=49\times30
$$

每个网格的 30 个输出由以下部分组成：

- Box 1：5 个值。
- Box 2：5 个值。
- Class Probabilities：20 个值。

整个前向传播：

~~~mermaid
flowchart TD
    A["Input [1,3,448,448]"] --> B["24 Conv Layers"]
    B --> C["Feature [1,1024,7,7]"]
    C --> D["Flatten [1,50176]"]
    D --> E["FC [1,4096]"]
    E --> F["FC [1,1470]"]
    F --> G["Reshape [1,7,7,30]"]
    G --> H["49 Grid Cells"]
    H --> I["98 Bounding Boxes"]
    H --> J["49 Class Probability Vectors"]
~~~

### 5.5 一个网格的输出示例

~~~python
# 一个网格的逻辑输出，VOC 20 类
prediction = [
    # Box 1: x, y, w, h, confidence
    0.40, 0.60, 0.25, 0.40, 0.70,

    # Box 2: x, y, w, h, confidence
    0.30, 0.70, 0.15, 0.30, 0.60,

    # Class probabilities: 20 classes
    0.70, 0.20, 0.10,
    0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00
]
~~~

这个示例恰好有：

$$
5+5+20=30
$$

个值。

如果数据集只有 3 个类别，那么每个网格输出：

$$
2\times5+3=13
$$

个值，而不是 30 个。

---

# 阶段 B：理解训练与推理（Understand Training & Inference）

## 6. 参数初始化与预训练

### 6.1 ImageNet 预训练

YOLOv1 并不是完全从随机初始化开始训练检测模型。

论文先使用 ImageNet 1000 类分类任务预训练卷积层：

1. 使用 Figure 3 中前 20 个卷积层。
2. 后接 Average Pooling。
3. 后接全连接分类层。
4. 在 ImageNet 1000 类数据集上训练。

论文报告预训练大约持续一周，ImageNet 2012 验证集上的单裁剪 Top-5 Accuracy 达到 88%。

### 6.2 转换为检测网络

在预训练网络基础上：

- 保留前 20 个卷积层。
- 增加 4 个卷积层。
- 增加 2 个全连接层。
- 新增加的卷积层和全连接层使用随机初始化权重。
- 将输入分辨率从 224×224 提升到 448×448。

论文没有明确给出所有新增层采用 Xavier、Kaiming 等哪一种具体初始化算法，因此不能自行将某种初始化方式写成论文事实。

### 6.3 为什么提高输入分辨率？

分类任务主要关注图像中是什么类别；检测任务还需要精确定位目标。

提高输入分辨率可以保留更多空间细节，帮助网络学习边界框位置和尺寸。

---

## 7. 正负样本分配：Responsible Predictor

YOLOv1 每个网格预测两个框，但一个 GT 应该由哪个框负责？

### 7.1 根据 IoU 选择 Responsible Predictor

假设一个网格中有一个 GT，两个预测框的 IoU 为：

| Predictor | IoU |
|---|---:|
| Box 1 | 0.80 |
| Box 2 | 0.56 |

那么：

$$
j^*=\arg\max_j IoU(B_{ij},GT_i)
$$

因此 Box 1 负责 GT。

只有 Responsible Predictor 计算坐标回归损失。

### 7.2 为什么只选择一个？

如果两个预测框都对同一个 GT 计算完整坐标损失，它们会被训练成相似的预测器，可能缺少预测偏好的分化。

YOLOv1 只让 IoU 最大的预测器负责该目标，使不同预测器在训练过程中逐渐形成不同的尺寸、宽高比或目标类型偏好。

这是 YOLOv1 的动态预测器分配机制，不是预先固定 Box 1 负责大目标、Box 2 负责小目标。

### 7.3 三个指示变量

完整 Loss 中有三个容易混淆的指示变量。

#### 网格级指标

$$
\mathbb{1}_i^{obj}=
\begin{cases}
1,&\text{网格 }i\text{ 中存在 GT}\\
0,&\text{网格 }i\text{ 中没有 GT}
\end{cases}
$$

它决定是否计算这个网格的分类损失。

#### 预测框级正样本指标

$$
\mathbb{1}_{ij}^{obj}=
\begin{cases}
1,&\text{预测器 }j\text{ 负责 GT}\\
0,&\text{否则}
\end{cases}
$$

它决定是否计算这个预测框的坐标损失和正样本 Confidence Loss。

#### 预测框级 No-Object 指标

论文公式使用了 $\mathbb{1}_{ij}^{noobj}$，但正文没有像前两个指标一样单独展开定义。结合作者 Darknet 参考实现，可以理解为：

- 每个网格的所有预测框最初都施加 No-Object Confidence 惩罚。
- 如果网格中有 GT，选出一个 Responsible Predictor。
- 该 Responsible Predictor 的 No-Object 惩罚被撤销，改为正样本 Confidence 监督。
- 同一个正样本网格中的其他预测框仍保留 No-Object Confidence 惩罚。

因此，在一个 GT 只由一个预测器负责的例子中：

$$
\mathbb{1}_{ij}^{noobj}=1-\mathbb{1}_{ij}^{obj}
$$

这不是说网格中有没有 GT，而是判断该预测框是否被选为负责预测器。

### 7.4 三种场景

| 场景 | 网格有 GT？ | Box 1 | Box 2 |
|---|---|---|---|
| A | 是，Box 1 负责 | obj=1，noobj=0 | obj=0，noobj=1 |
| B | 否 | obj=0，noobj=1 | obj=0，noobj=1 |
| C | 是，Box 2 负责 | obj=0，noobj=1 | obj=1，noobj=0 |

这张表是理解 YOLOv1 Loss 的关键。

---

## 8. Loss Function：完整公式与变量含义

YOLOv1 使用 Sum of Squared Error（SSE），也就是平方误差。

总损失分为四个部分：

1. Bounding Box 坐标损失。
2. 负责预测框的 Confidence 损失。
3. No-Object Confidence 损失。
4. 网格级分类损失。

### 8.1 Bounding Box 坐标损失

对负责预测器计算：

$$
\begin{aligned}
L_{box}=\sum_i\sum_j\mathbb{1}_{ij}^{obj}
\Big[&
(x_{ij}^*-\hat x_{ij})^2\\
&+(y_{ij}^*-\hat y_{ij})^2\\
&+(\sqrt{w_{ij}^*}-\sqrt{\hat w_{ij}})^2\\
&+(\sqrt{h_{ij}^*}-\sqrt{\hat h_{ij}})^2
\Big]
\end{aligned}
$$

为什么宽度和高度使用平方根？

假设两个目标：

- 大目标宽度 200 px，预测偏差 5 px。
- 小目标宽度 10 px，预测偏差 5 px。

相同的绝对误差对小目标造成的定位影响更大。

直接对宽高使用平方误差，会让大目标的误差数值更容易占据较大权重。

使用平方根可以部分缓解这个问题：

$$
(\sqrt{w^*}-\sqrt{\hat w})^2
$$

但它并不能完全解决不同尺度目标的 IoU 误差不一致问题。

### 8.2 负责预测框的 Confidence Loss

对于 Responsible Predictor：

$$
C_{ij}^*=IoU(B_{ij},GT_i)
$$

预测值为：

$$
\hat C_{ij}
$$

损失：

$$
L_{conf,obj}=
\sum_i\sum_j
\mathbb{1}_{ij}^{obj}
(C_{ij}^*-\hat C_{ij})^2
$$

负责预测框的目标 Confidence 就是它与 GT 的 IoU。

### 8.3 No-Object Confidence Loss

对于没有负责任何 GT 的预测框，Confidence 目标为 0：

$$
C_{ij}^*=0
$$

对应损失：

$$
L_{conf,noobj}=
\sum_i\sum_j
\mathbb{1}_{ij}^{noobj}
(0-\hat C_{ij})^2
$$

这一项乘以：

$$
\lambda_{noobj}=0.5
$$

降低背景预测框对整体梯度的影响。

### 8.4 分类损失

每个网格只有一组类别概率，所以分类损失按网格计算一次：

$$
L_{class}=
\sum_i\mathbb{1}_i^{obj}
\sum_c
(p_i^*(c)-\hat p_i(c))^2
$$

其中：

- $p_i^*(c)$：GT 类别 One-Hot 标签。
- $\hat p_i(c)$：网络预测的条件类别概率。
- $\mathbb{1}_i^{obj}$：当前网格是否包含 GT。

没有目标的网格不计算分类损失。

### 8.5 完整 Loss

$$
\boxed{
L=\lambda_{coord}L_{box}
+L_{conf,obj}
+\lambda_{noobj}L_{conf,noobj}
+L_{class}
}
$$

论文使用：

$$
\lambda_{coord}=5,\quad
\lambda_{noobj}=0.5
$$

展开后：

$$
\begin{aligned}
L={}&
5\sum_i\sum_j\mathbb{1}_{ij}^{obj}
\left[
(x_{ij}^*-\hat x_{ij})^2+
(y_{ij}^*-\hat y_{ij})^2
\right]\\
&+5\sum_i\sum_j\mathbb{1}_{ij}^{obj}
\left[
(\sqrt{w_{ij}^*}-\sqrt{\hat w_{ij}})^2+
(\sqrt{h_{ij}^*}-\sqrt{\hat h_{ij}})^2
\right]\\
&+\sum_i\sum_j\mathbb{1}_{ij}^{obj}
(C_{ij}^*-\hat C_{ij})^2\\
&+0.5\sum_i\sum_j\mathbb{1}_{ij}^{noobj}
(0-\hat C_{ij})^2\\
&+\sum_i\mathbb{1}_i^{obj}
\sum_c(p_i^*(c)-\hat p_i(c))^2
\end{aligned}
$$

这里：

- $\mathbb{1}_{ij}^{obj}$：预测框级别，是否负责 GT。
- $\mathbb{1}_{ij}^{noobj}$：预测框级别，是否不负责 GT。
- $\mathbb{1}_{i}^{obj}$：网格级别，是否有 GT。

**不要把网格级的 $\mathbb{1}_{i}^{obj}$ 和预测框级的 $\mathbb{1}_{ij}^{obj}$ 混为一谈。**

---

## 9. 完整数值例子：使用 30 个输出值计算 Loss

这一节使用具体数值，把输出、GT、指示变量和每一项 Loss 对应起来。

### 9.1 网络输出

假设数据集有 VOC 的 20 个类别，但为了方便计算，只让前三个类别输出非零值：

~~~python
# 一个网格的逻辑输出：30 个数

# Box 1: x, y, w, h, confidence
[0.40, 0.60, 0.25, 0.40, 0.70,

 # Box 2: x, y, w, h, confidence
 0.30, 0.70, 0.15, 0.30, 0.60,

 # Class probabilities: 20 classes
 0.70, 0.20, 0.10,
 0.00, 0.00, 0.00, 0.00, 0.00,
 0.00, 0.00, 0.00, 0.00, 0.00,
 0.00, 0.00, 0.00, 0.00, 0.00,
 0.00, 0.00]
~~~

我们约定：

- 类别 0：Cat。
- 类别 1：Dog。
- 类别 2：Person。
- 其余 17 类概率为 0。

两个预测框：

$$
Box_1=(0.40,0.60,0.25,0.40,0.70)
$$

$$
Box_2=(0.30,0.70,0.15,0.30,0.60)
$$

注意：0.70 和 0.60 是两个预测框自己的 Confidence 输出；0.70、0.20、0.10 是共享的类别条件概率。它们不是同一组概率。

### 9.2 场景 A：网格有一个 Cat，Box 1 负责 GT

设 GT：

$$
GT=(0.40,0.60,0.20,0.40)
$$

GT 类别为 Cat，即类别 0。

先计算 Box 1 的 IoU。两个框中心相同，GT 完全位于 Box 1 内：

$$
Area_{GT}=0.20\times0.40=0.08
$$

$$
Area_{Box1}=0.25\times0.40=0.10
$$

$$
IoU_1=\frac{0.08}{0.10}=0.8
$$

Box 2 的中心也在同一个网格内，换算到图片坐标后，Box 2 完全位于 GT 内：

$$
IoU_2=\frac{0.15\times0.30}{0.20\times0.40}
=0.5625
$$

因此：

$$
IoU_1>IoU_2
$$

Box 1 负责 GT。

#### 第一步：确定指示变量

$$
\mathbb{1}_{i}^{obj}=1
$$

$$
\mathbb{1}_{i1}^{obj}=1,\quad
\mathbb{1}_{i1}^{noobj}=0
$$

$$
\mathbb{1}_{i2}^{obj}=0,\quad
\mathbb{1}_{i2}^{noobj}=1
$$

#### 第二步：计算 Box 坐标损失

只有 Box 1 计算坐标损失：

$$
\begin{aligned}
L_{box}={}&
(0.40-0.40)^2\\
&+(0.60-0.60)^2\\
&+(\sqrt{0.20}-\sqrt{0.25})^2\\
&+(\sqrt{0.40}-\sqrt{0.40})^2
\end{aligned}
$$

$$
L_{box}\approx0.0027864
$$

乘以坐标权重：

$$
5L_{box}\approx0.013932
$$

Box 2 不计算坐标损失，因为它没有负责 GT。

#### 第三步：计算负责预测框的 Confidence Loss

Box 1：

$$
C_1^*=IoU_1=0.8
$$

网络预测：

$$
\hat C_1=0.70
$$

因此：

$$
L_{conf,obj}=(0.8-0.7)^2=0.01
$$

#### 第四步：计算非负责预测框的 Confidence Loss

Box 2 不负责 GT，目标 Confidence 为 0：

$$
L_{conf,noobj}=0.5(0-0.60)^2
$$

$$
L_{conf,noobj}=0.18
$$

#### 第五步：计算分类损失

GT 类别为 Cat，One-Hot：

$$
P^*=[1,0,0,0,\ldots,0]
$$

预测：

$$
\hat P=[0.7,0.2,0.1,0,\ldots,0]
$$

所以：

$$
\begin{aligned}
L_{class}={}&(1-0.7)^2\\
&+(0-0.2)^2\\
&+(0-0.1)^2
\end{aligned}
$$

$$
L_{class}=0.09+0.04+0.01=0.14
$$

#### 第六步：汇总

$$
\begin{aligned}
L={}&5L_{box}
+L_{conf,obj}\\
&+L_{conf,noobj}
+L_{class}
\end{aligned}
$$

$$
L=0.013932+0.01+0.18+0.14
$$

$$
\boxed{L\approx0.343932}
$$

### 9.3 场景 B：这个网格没有任何 GT 中心点

网络输出完全不变，但当前网格没有 GT。

因此：

$$
\mathbb{1}_{i}^{obj}=0
$$

两个预测框都不负责 GT：

$$
\mathbb{1}_{i1}^{obj}=0,\quad
\mathbb{1}_{i2}^{obj}=0
$$

$$
\mathbb{1}_{i1}^{noobj}=1,\quad
\mathbb{1}_{i2}^{noobj}=1
$$

**Box Loss：**

$$
L_{box}=0
$$

**Positive Confidence Loss：**

$$
L_{conf,obj}=0
$$

**No-Object Confidence Loss：**

$$
\begin{aligned}
L_{conf,noobj}
&=0.5[(0-0.70)^2+(0-0.60)^2]\\
&=0.5(0.49+0.36)\\
&=\boxed{0.425}
\end{aligned}
$$

**Classification Loss：**

$$
L_{class}=0
$$

因为网格没有 GT，不计算类别损失。

最终：

$$
\boxed{L=0.425}
$$

这个场景说明：背景网格不需要训练框的位置和类别，只需要让所有预测框的 Confidence 逐渐接近 0。

### 9.4 场景 C：GT 类别为序号 1，Box 2 与 GT 的 IoU 更大

仍然使用同一组网络输出：

$$
Box_1=(0.40,0.60,0.25,0.40,0.70)
$$

$$
Box_2=(0.30,0.70,0.15,0.30,0.60)
$$

现在设 GT：

$$
GT=(0.32,0.68,0.16,0.32)
$$

GT 类别为序号 1，也就是 Dog。

这里 x、y 是相对于网格的坐标，w、h 是相对于整张图片的尺寸。计算 IoU 时需要将两个框的中心坐标还原到同一个图片坐标系：

$$
x_{image}=\frac{col+x_{grid}}{7}
$$

$$
y_{image}=\frac{row+y_{grid}}{7}
$$

因为两个预测框和 GT 都属于同一个网格，以下重叠关系可以直接通过相对位置判断。

**计算 Box 1 的 IoU：**

GT 完全位于 Box 1 内：

$$
IoU_1=
\frac{0.16\times0.32}{0.25\times0.40}
=0.512
$$

**计算 Box 2 的 IoU：**

Box 2 完全位于 GT 内：

$$
IoU_2=
\frac{0.15\times0.30}{0.16\times0.32}
$$

$$
IoU_2\approx0.878906
$$

因此：

$$
IoU_2>IoU_1
$$

Box 2 负责 GT。

#### 第一步：指示变量

$$
\mathbb{1}_{i}^{obj}=1
$$

$$
\mathbb{1}_{i1}^{obj}=0,\quad
\mathbb{1}_{i1}^{noobj}=1
$$

$$
\mathbb{1}_{i2}^{obj}=1,\quad
\mathbb{1}_{i2}^{noobj}=0
$$

#### 第二步：Box 坐标损失

只有 Box 2 计算：

$$
\begin{aligned}
L_{box}={}&
(0.32-0.30)^2\\
&+(0.68-0.70)^2\\
&+(\sqrt{0.16}-\sqrt{0.15})^2\\
&+(\sqrt{0.32}-\sqrt{0.30})^2
\end{aligned}
$$

$$
L_{box}\approx0.001284
$$

乘以坐标权重：

$$
5L_{box}\approx0.006420
$$

#### 第三步：负责预测框 Confidence Loss

Box 2 的目标 Confidence：

$$
C_2^*=IoU_2\approx0.878906
$$

预测 Confidence：

$$
\hat C_2=0.60
$$

因此：

$$
L_{conf,obj}=(0.878906-0.60)^2
$$

$$
L_{conf,obj}\approx0.077789
$$

#### 第四步：非负责预测框 Confidence Loss

Box 1 不负责 GT：

$$
L_{conf,noobj}=0.5(0-0.70)^2
$$

$$
L_{conf,noobj}=0.245
$$

#### 第五步：分类损失

GT 类别为 Dog，即类别序号 1。

One-Hot：

$$
P^*=[0,1,0,0,\ldots,0]
$$

预测：

$$
\hat P=[0.70,0.20,0.10,0,\ldots,0]
$$

因此：

$$
\begin{aligned}
L_{class}={}&(0-0.70)^2\\
&+(1-0.20)^2\\
&+(0-0.10)^2
\end{aligned}
$$

$$
L_{class}=0.49+0.64+0.01=1.14
$$

注意：分类损失是网格级的，只计算一次，并不是只对 Box 2 计算。

#### 第六步：汇总

$$
\begin{aligned}
L={}&0.006420\\
&+0.077789\\
&+0.245\\
&+1.14
\end{aligned}
$$

$$
\boxed{L\approx1.469209}
$$

### 9.5 三个场景的对比

| 场景 | GT | Responsible Predictor | Box Loss | Positive Conf | No-Object Conf | Class Loss |
|---|---|---|---:|---:|---:|---:|
| A | Cat | Box 1 | 0.013932 | 0.010000 | 0.180000 | 0.140000 |
| B | 无 | 无 | 0 | 0 | 0.425000 | 0 |
| C | Dog | Box 2 | 0.006420 | 0.077789 | 0.245000 | 1.140000 |

这些数值只计算了示例网格的 Loss。真实训练时，还需要累加整张图片中其他网格的损失，并对 Batch 中的样本进行聚合。

---

## 10. 完整训练流程（Training Pipeline）

### 10.1 训练过程

~~~mermaid
flowchart TD
    A["输入图片与 GT"] --> B["Data Augmentation"]
    B --> C["YOLOv1 Forward"]
    C --> D["输出 7×7×30"]
    A --> E["GT 中心点分配 Grid"]
    E --> F["计算预测框与 GT 的 IoU"]
    D --> F
    F --> G["选择 Responsible Predictor"]
    G --> H["构造 obj / noobj 指示变量"]
    H --> I["计算 Box Loss"]
    H --> J["计算 Confidence Loss"]
    H --> K["计算 Class Loss"]
    I --> L["汇总 Loss"]
    J --> L
    K --> L
    L --> M["Backpropagation"]
    M --> N["Optimizer 更新参数"]
~~~

### 10.2 训练配置

论文报告的主要检测训练设置：

| 参数 | 设置 |
|---|---:|
| Epoch | 约 135 |
| Batch Size | 64 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| $\lambda_{coord}$ | 5 |
| $\lambda_{noobj}$ | 0.5 |
| Dropout | 0.5 |
| 输入尺寸 | 448×448 |

学习率策略：

| 训练阶段 | 学习率 |
|---|---:|
| 初始阶段 | 从 0.001 缓慢升至 0.01 |
| 接下来 75 Epoch | 0.01 |
| 接下来 30 Epoch | 0.001 |
| 最后 30 Epoch | 0.0001 |

论文指出，直接从较大学习率开始训练容易因为梯度不稳定而发散，因此采用了学习率预热。

### 10.3 数据增强

论文使用：

- 随机缩放。
- 随机平移，最大约为原始图像尺寸的 20%。
- HSV 空间中的曝光和饱和度变化，调整因子最高约 1.5。

这些增强让模型学习对目标位置、尺寸和颜色变化更鲁棒的表示。

### 10.4 训练数据

论文使用 PASCAL VOC 2007 和 VOC 2012 的训练集与验证集进行训练。

当报告 VOC 2012 测试结果时，还将 VOC 2007 测试数据加入训练数据。

需要区分：

- ImageNet：用于卷积 Backbone 预训练。
- VOC 2007 / VOC 2012：用于检测训练与评估。

论文没有给出完整检测训练阶段的总耗时。可以确认的是 ImageNet 预训练约一周，不应将它误写为 YOLOv1 整个检测训练过程只需一周。

---

## 11. 推理过程与后处理（Inference Pipeline）

### 11.1 输入预处理

论文中的推理流程首先将图片 Resize 到：

$$
448\times448
$$

随后只需要一次网络前向传播。

### 11.2 预测框解码

网络输出：

$$
Y\in\mathbb{R}^{1\times7\times7\times30}
$$

假设第 $i$ 个网格的列索引为 $c$、行索引为 $r$。

网络预测：

$$
(x,y,w,h)
$$

其中 x、y 是网格内部坐标。

转换为整张图片归一化中心坐标：

$$
x_c=\frac{c+x}{7}
$$

$$
y_c=\frac{r+y}{7}
$$

宽高已经相对于整张图片归一化，因此：

$$
w_c=w,\quad h_c=h
$$

转换为左上角和右下角：

$$
x_1=x_c-\frac{w_c}{2}
$$

$$
y_1=y_c-\frac{h_c}{2}
$$

$$
x_2=x_c+\frac{w_c}{2}
$$

$$
y_2=y_c+\frac{h_c}{2}
$$

最后乘以原始图片宽高，得到像素坐标。

### 11.3 计算每个类别的检测分数

每个预测框都有一个 Confidence，每个网格共享一组类别概率。

$$
Score(c)=Confidence\times P(Class_c\mid Object)
$$

例如：

$$
Confidence=0.8
$$

$$
P(Cat\mid Object)=0.9
$$

则：

$$
Score(Cat)=0.8\times0.9=0.72
$$

一个预测框可以得到多个类别分数，后处理根据类别分数进行阈值筛选。

### 11.4 NMS

YOLOv1 仍然需要 NMS。

论文设置：

$$
7\times7\times2=98
$$

个候选框/图片。

典型 NMS 流程：

1. 对某个类别的候选框按类别分数从高到低排序。
2. 保留当前最高分框。
3. 计算它与其他候选框的 IoU。
4. 将 IoU 超过 NMS 阈值的低分框抑制。
5. 继续处理剩余候选框。

例如：

- Box A 分数 0.90。
- Box B 分数 0.80。
- 两者属于同一类别，IoU=0.85。
- NMS 阈值为 0.50。

保留 Box A，抑制 Box B。

论文指出 NMS 对 YOLOv1 并不像对 R-CNN 或 DPM 那样关键，但可以处理大目标或靠近网格边界的目标被多个网格重复检测的问题。论文报告 NMS 可带来约 2%–3% 的 mAP 增益。

### 11.5 Training 与 Inference 对比

| 项目 | Training | Inference |
|---|---|---|
| 输入 | 图片和 GT | 只有图片 |
| Forward | CNN + FC | CNN + FC |
| 输出 | 7×7×30 | 7×7×30 |
| GT 网格分配 | 需要 | 不需要 |
| Responsible Predictor | 根据 IoU 选择 | 不需要 |
| Loss | 计算 | 不计算 |
| Backpropagation | 需要 | 不需要 |
| Box Decode | 用于匹配与 Loss | 用于生成检测框 |
| Confidence | 与 IoU 监督目标比较 | 直接作为预测分数的一部分 |
| NMS | 不需要 | 需要 |

---

## 12. 可解释性与特征可视化（Interpretability）

### 12.1 Figure 2：Grid、Bounding Box 和类别概率图

原论文 Figure 2 展示了：

1. 输入图片上的网格。
2. 每个网格产生的多个预测框及 Confidence。
3. 网格级类别概率图。
4. 最终检测结果。

这张图是对 YOLOv1 输出机制的概念性可视化。

它不是 CNN 某一层的 Feature Map，也不是 Grad-CAM 或 Attention Heatmap。

### 12.2 Figure 3：网络结构图

Figure 3 展示 24 个卷积层、2 个全连接层以及特征图空间尺寸变化。

它用于解释网络架构，而不是展示模型学到的激活区域。

### 12.3 Figure 4：检测错误类型分析

论文比较 YOLO 与 Fast R-CNN 的预测错误分布：

| 错误类型 | Fast R-CNN | YOLO |
|---|---:|---:|
| Correct | 71.6% | 65.5% |
| Localization | 8.6% | 19.0% |
| Similar | 4.3% | 6.75% |
| Other | 1.9% | 4.0% |
| Background | 13.6% | 4.75% |

该统计基于论文定义的每类 Top-N 检测结果。

它说明 YOLO 的背景误检比例较低，但定位错误比例更高。这与论文对 YOLO 的整体评价一致：模型能够较好地利用整图上下文，但边界框定位精度仍有不足。

### 12.4 论文没有提供什么？

原始 YOLOv1 论文没有提供现代视觉论文中常见的：

- Grad-CAM。
- Attention Map。
- 多层 Feature Map 热力图。
- 特征嵌入 t-SNE。
- Backbone 不同层的激活可视化。

如果要进一步分析 YOLOv1 学到了什么，可以自行提取卷积层 Feature Map、使用 Grad-CAM，或观察不同网格的类别和 Confidence 输出。但这些属于额外复现实验，不能说成论文原有结果。

---

# 阶段 C：评价模型（Evaluate the Model）

## 13. 数据集与实验设置

### 13.1 PASCAL VOC

YOLOv1 主要在 PASCAL VOC 2007 和 VOC 2012 上训练与评估。

VOC 检测任务包含 20 个类别，因此论文的输出通道为：

$$
2\times5+20=30
$$

### 13.2 预训练数据

模型的卷积 Backbone 使用 ImageNet 1000 类分类数据预训练。

因此，论文实验不是完全从随机初始化开始的检测训练。

### 13.3 测试条件

论文报告的速度数据是在 NVIDIA Titan X GPU 上测量，推理没有使用 Batch Processing。

历史速度不能直接与当前 GPU 上的 YOLO11 / YOLO26 FPS 比较，因为硬件、软件栈、输入尺寸、精度模式和测试流程均不同。

---

## 14. 性能评估与实验结果

### 14.1 VOC 2007 实时检测器对比

论文报告：

| Detector | VOC 2007 mAP | FPS |
|---|---:|---:|
| Fast YOLO | 52.7 | 155 |
| YOLO | 63.4 | 45 |

YOLOv1 的主要价值是以一次网络前向传播实现实时检测，在当时显著简化了检测流程。

### 14.2 精度与速度的取舍

Fast YOLO：

- 采用更小的网络。
- 只有 9 个卷积层。
- 推理速度达到 155 FPS。
- mAP 为 52.7。

标准 YOLO：

- 使用 24 个卷积层。
- mAP 达到 63.4。
- 推理速度为 45 FPS。

这体现了模型容量、精度和速度之间的权衡。

### 14.3 训练耗时

论文明确报告 ImageNet 预训练约一周，但没有给出足以严格复现的检测训练总耗时、单 Epoch 时间或完整训练吞吐量。

因此，不应根据 135 Epoch 直接推算一个确定的总训练时间。

---

## 15. 与前代检测方法相比的创新

YOLOv1 的主要贡献不是简单地把 CNN 用于检测，而是重新定义检测流程：

| 比较维度 | 传统多阶段检测流程 | YOLOv1 |
|---|---|---|
| 候选区域 | 需要 Region Proposal 等步骤 | 不需要单独生成候选区域 |
| 特征提取 | 可能对多个区域分别运行 | 整张图片一次提取 |
| 分类 | 区域级分类 | 网格级条件类别预测 |
| 定位 | 独立的区域回归 | 与分类共同训练 |
| 全局上下文 | 局部区域可能缺少全局信息 | 网络直接看到整张图片 |
| 后处理 | 通常依赖 NMS | 仍使用 NMS，但候选框较少 |
| 优化 | 多组件流程 | 单一网络端到端训练 |

### 15.1 YOLOv1 为什么能减少背景误检？

网络训练时始终看到整张图片，能够利用背景和目标之间的上下文关系。

论文指出，与 Fast R-CNN 相比，YOLO 的背景错误更少。

### 15.2 YOLOv1 为什么定位误差较多？

论文的错误分析显示，YOLO 的 Localization 错误比例明显高于 Fast R-CNN。

原因包括：

- 网格空间约束较强。
- 每个网格只预测两个框。
- 最终特征图空间分辨率较低。
- 使用 SSE 近似优化检测目标，不能直接等价于最大化 mAP。
- 小目标的定位误差对 IoU 影响更明显。

---

## 16. 模型不足与失败案例

### 16.1 网格空间约束

每个网格只有两个 Bounding Box Predictor。

如果多个目标的中心落在同一个网格，预测能力受到限制。

### 16.2 每个网格只有一组类别概率

两个 Bounding Box 共用一个类别概率向量。

这意味着模型难以在同一网格中独立表达多个不同类别的目标。

这是 YOLOv1 对密集目标和相邻不同类别目标表现不佳的重要原因。

### 16.3 小目标与密集目标

论文指出，YOLOv1 对成群的小目标检测效果不好，例如鸟群。

原因包括：

- 网格划分较粗。
- 每个网格预测框数量有限。
- 最终特征图分辨率较低。
- 多个小目标可能集中在相同网格或相邻网格。

### 16.4 特殊宽高比和目标形状

YOLOv1 直接从数据学习边界框坐标，没有 Anchor 尺寸先验。

当目标具有训练数据中少见的宽高比或配置时，模型可能难以泛化。

### 16.5 坐标损失与检测指标不完全一致

YOLOv1 使用 SSE 作为主要优化目标。

但检测评价通常依赖 IoU 和 AP。坐标 SSE 下降不一定意味着 AP 按相同比例提升。

平方根宽高只是在一定程度上缓解大目标和小目标的误差尺度差异，并不能彻底解决这个问题。

### 16.6 主要错误类型

结合论文 Figure 4：

- YOLO 的 Localization 错误更多。
- YOLO 的 Background 错误更少。
- 模型更擅长利用整图上下文减少背景误检，但定位精度仍有明显改进空间。

---

# 阶段 D：复现与应用（Reproduce & Apply）

## 17. 官方源码与论文结构的对应

YOLOv1 的作者代码位于 [pjreddie/darknet](https://github.com/pjreddie/darknet)。

建议重点阅读：

| 论文概念 | Darknet 文件 | 主要内容 |
|---|---|---|
| 网络配置 | `cfg/yolov1.cfg` | 历史 YOLOv1 配置 |
| Tiny 配置 | `cfg/yolov1-tiny.cfg` | Tiny 变体 |
| Detection Layer | `src/detection_layer.c` | 正负 Confidence、责任预测器、坐标与类别梯度 |
| IoU | `src/box.c` | Box Intersection、Union、IoU |
| 配置解析 | `src/parser.c` | Detection Layer 配置解析 |
| 卷积实现 | `src/convolutional_layer.c` | 卷积层前向与反向计算 |
| 全连接实现 | `src/connected_layer.c` | 全连接层前向与反向计算 |

### 17.1 Detection Layer 中的关键逻辑

Darknet 的 `src/detection_layer.c` 中，训练过程可以概括为：

~~~python
for each_grid_cell:
    # 默认所有预测框都受到 no-object 约束
    for predictor in predictors:
        confidence_target = 0
        apply_noobject_confidence_loss()

    if grid_cell_has_no_ground_truth:
        continue

    # 类别损失是网格级的，只计算一次
    calculate_classification_loss()

    # 计算该网格所有预测框与 GT 的 IoU
    best_predictor = argmax_iou(predicted_boxes, ground_truth)

    # 将最佳预测器从 no-object 监督切换为 object 监督
    remove_noobject_loss(best_predictor)

    confidence_target = IoU(best_predictor, ground_truth)
    calculate_positive_confidence_loss()

    # 只有最佳预测器计算坐标回归
    calculate_box_coordinate_loss(best_predictor)
~~~

这段伪代码解释了为什么：

- 背景网格的两个预测框都计算 No-Object Loss。
- 正样本网格中，未被选中的预测框仍然计算 No-Object Loss。
- 只有负责预测框计算坐标损失。
- 分类损失按网格计算一次。

源码中还有一个实现细节：如果该网格所有候选框的 IoU 都为 0，参考实现会退回到 RMSE，选择距离误差更小的预测框。这个行为属于该实现的细节，不应误认为论文中单独提出的主要算法。

### 17.2 IoU 的源码计算

`src/box.c` 中的 IoU 逻辑可以概括为：

$$
IoU=\frac{Intersection}{Union}
$$

其中：

$$
Union=Area(A)+Area(B)-Intersection
$$

源码通过两个框中心坐标和宽高计算重叠宽度、重叠高度，再计算交集面积和并集面积。

参考：

- [Darknet box.c](https://github.com/pjreddie/darknet/blob/master/src/box.c)
- [Darknet detection_layer.c](https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c)

---

## 18. 一个非常重要的版本差异：论文设置不等于 Darknet 当前配置

这是复现 YOLOv1 时特别容易出错的地方。

### 18.1 论文中的标准 YOLOv1

| 参数 | 论文 |
|---|---:|
| 输入 | 448×448 |
| Grid | 7×7 |
| Box Predictors | 2 |
| 类别 | 20 |
| 每网格输出 | 30 |
| 总输出 | 1470 |
| 网络 | 24 Conv + 2 FC |

### 18.2 Darknet 仓库中的 `cfg/yolov1.cfg`

当前作者仓库中的这个配置包含：

| 参数 | 配置值 |
|---|---:|
| 输入 | 448×448 |
| Grid | 7×7 |
| classes | 20 |
| coords | 4 |
| num | 3 |
| 每网格输出 | $3\times5+20=35$ |
| 总输出 | $7\times7\times35=1715$ |

并且配置中包含：

- Batch Normalization。
- Local Layer。
- Dropout。
- 一个输出为 1715 的 Connected Layer。
- Detection Layer。

因此，不能把当前 `cfg/yolov1.cfg` 的全部配置直接当作论文 Figure 3 的精确网络结构。

### 18.3 Darknet 的 `cfg/yolov1-tiny.cfg`

这个配置使用：

$$
num=2
$$

$$
7\times7\times(2\times5+20)=1470
$$

它虽然也得到 1470 个输出值，但网络结构是 Tiny 变体，不能因为输出维度相同，就认为它等于论文的完整 YOLOv1。

### 18.4 复现时应该怎样选择？

建议明确区分三个目标：

| 复现目标 | 采用资料 | 说明 |
|---|---|---|
| 理解论文算法 | 原论文 | 采用 $S=7,B=2,C=20$ |
| 复现作者历史配置 | Darknet `yolov1.cfg` | 接受其 $num=3$ 等实现差异 |
| 研究轻量变体 | Darknet `yolov1-tiny.cfg` | 独立分析 Tiny 结构 |

不要把论文的 7×7×30 输出、Darknet 配置的 1715 输出以及 Tiny 的结构拼接成一个不存在的模型。

---

## 19. 复现时的检查清单

### 19.1 网络结构

- [ ] 输入尺寸是否为 448×448？
- [ ] Grid 是否为 7×7？
- [ ] 每个网格预测框数量是否与目标版本一致？
- [ ] 类别数是否正确？
- [ ] 输出维度是否满足 $S\times S\times(B\times5+C)$？
- [ ] 全连接层输入维度是否正确？
- [ ] 最终输出是否使用线性激活？

### 19.2 训练机制

- [ ] 是否使用 ImageNet 预训练卷积权重？
- [ ] 新增卷积和全连接层是否正确初始化？
- [ ] GT 是否按中心点分配到网格？
- [ ] 是否根据 IoU 选择 Responsible Predictor？
- [ ] 只有 Responsible Predictor 是否计算坐标损失？
- [ ] No-Object Confidence 目标是否正确？
- [ ] 分类损失是否按网格计算一次？
- [ ] 是否使用 $\lambda_{coord}=5$ 和 $\lambda_{noobj}=0.5$？

### 19.3 推理机制

- [ ] 网格坐标是否正确解码？
- [ ] 宽高是否按整张图片归一化？
- [ ] 类别分数是否乘以 Box Confidence？
- [ ] 是否执行置信度过滤？
- [ ] 是否执行 NMS？
- [ ] 是否将坐标还原到原图？

---

## 20. 总结：YOLOv1 最值得掌握的知识

YOLOv1 是现代单阶段目标检测的重要起点。理解它，重点不是记住 24 层卷积，而是掌握以下机制。

### 20.1 检测问题被统一为整图回归

$$
Image\rightarrow Grid\rightarrow Boxes+Classes
$$

一个网络一次前向传播完成特征提取、位置预测和类别预测。

### 20.2 Grid 决定目标由谁负责

目标中心点所在的网格负责预测该目标。每个网格预测两个框，但只有一个预测器负责一个 GT。

### 20.3 Confidence 同时考虑目标存在性与定位质量

$$
Confidence=P(Object)\times IoU
$$

训练时负责预测框的目标 Confidence 是当前预测框与 GT 的 IoU；推理时使用网络预测的 Confidence 乘以条件类别概率。

### 20.4 Loss 由四个部分构成

$$
L=
5L_{box}
+L_{conf,obj}
+0.5L_{conf,noobj}
+L_{class}
$$

- Box Loss：只训练负责预测器。
- Positive Confidence Loss：目标为 IoU。
- No-Object Confidence Loss：目标为 0。
- Classification Loss：每个正样本网格计算一次。

### 20.5 YOLOv1 的主要限制推动了后续发展

- 网格约束和有限预测框数量。
- 每个网格只有一组类别概率。
- 小目标和密集目标检测不足。
- 直接回归框尺寸，对特殊宽高比泛化不足。
- 定位精度与检测指标之间存在差距。

这些问题也为后续 Anchor-Based、Anchor-Free、多尺度特征融合、改进标签分配和更合理的定位损失设计提供了研究方向。

---

## 参考资料

### 原始论文

1. [Joseph Redmon et al. — You Only Look Once: Unified, Real-Time Object Detection, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
2. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

### 作者代码仓库

3. [pjreddie/darknet](https://github.com/pjreddie/darknet)
4. [YOLOv1 Darknet 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)
5. [YOLOv1 Tiny 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg)
6. [Detection Layer 实现](https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c)
7. [Bounding Box 与 IoU 实现](https://github.com/pjreddie/darknet/blob/master/src/box.c)
8. [Darknet 配置解析](https://github.com/pjreddie/darknet/blob/master/src/parser.c)

### 后续模型对比

9. [YOLOv2 / YOLO9000](https://arxiv.org/abs/1612.08242)
10. [Ultralytics YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)
11. [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)
