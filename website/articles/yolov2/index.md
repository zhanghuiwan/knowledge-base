---
title: YOLOv2 与 YOLO9000：Anchor、Darknet-19、Passthrough 与联合训练
description: 结合 YOLO9000 原论文与 pjreddie/darknet 官方配置、源码，深入解析 YOLOv2 的 Anchor 参数化、K-means 聚类、Darknet-19、Passthrough、Region Loss、多尺度训练，以及 YOLO9000 的 WordTree 和检测分类联合训练。
---

# YOLOv2 与 YOLO9000：Anchor、Darknet-19、Passthrough 与联合训练

## 阅读说明

本文以 Joseph Redmon 和 Ali Farhadi 的 CVPR 2017 论文 **YOLO9000: Better, Faster, Stronger** 为主要研究对象，并结合作者的 [pjreddie/darknet](https://github.com/pjreddie/darknet) 仓库，分析 YOLOv2 的网络结构、训练机制和实际实现。

建议先阅读知识库中的 [YOLOv1：网络结构、Confidence、Loss 与完整训练推理解析](/articles/yolov1/)。YOLOv2 延续了 YOLOv1 的单阶段、整图检测思想，但在输出结构、Bounding Box 参数化、特征提取网络和训练策略上进行了系统升级。

需要先区分两个名称：

- **YOLOv2**：检测器本身的改进，重点是 Anchor Boxes、Dimension Clusters、Direct Location Prediction、Passthrough、Darknet-19 和 Multi-Scale Training。
- **YOLO9000**：在 YOLOv2 检测器基础上，进一步使用 WordTree 和检测/分类联合训练，使模型能够预测超过 9000 个类别。

**资料边界与版本说明：**

1. 论文中的 YOLOv2 使用 5 个 Anchor Boxes。以 VOC 20 类为例，输出通道数为 $5\times(5+20)=125$；以 COCO 80 类为例，输出通道数为 $5\times(5+80)=425$。
2. 论文介绍了 $416\times416$ 输入和 $13\times13$ 输出特征图，但 YOLOv2 通过 Multi-Scale Training 学习多种输入分辨率，论文实验也使用 288、352、416、480 和 544 等尺寸。
3. 作者仓库当前 COCO 配置 **cfg/yolov2.cfg** 默认测试尺寸为 608，VOC 配置 **cfg/yolov2-voc.cfg** 默认尺寸为 416。这些配置不能与论文某一个实验输入尺寸混为一谈。
4. YOLO9000 使用 9418 个 WordTree 类别节点和 3 个 Anchor，因此检测头输出通道数为 $3\times(5+9418)=28269$。
5. 论文、Darknet 配置和源码分别承担不同的证据作用。论文用于说明方法与实验结论，配置用于确认参数，源码用于确认实际张量处理和训练逻辑。

原始论文：[YOLO9000: Better, Faster, Stronger（CVPR 2017）](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)。

---

# 阶段 A：理解模型（Understand the Model）

## 1. 研究背景与核心思想

### 1.1 YOLOv1 的问题

YOLOv1 将目标检测统一为一个 CNN 的整图回归任务，输入整张图片，直接输出 Bounding Box、Confidence 和类别概率。

这种设计带来了实时检测能力，但论文分析发现 YOLOv1 相比当时的 Faster R-CNN 等方法仍有两个明显问题：

1. **Localization Error 较多**：边界框位置和尺寸预测不够准确。
2. **Recall 较低**：模型能够提出的有效目标候选较少。

YOLOv1 的每个网格只预测两个 Bounding Box，所有 Box 共享一组类别概率，且最后使用全连接层输出固定的 $7\times7\times30$ 结果。这些设计限制了模型的候选框数量、空间表达和输入尺寸灵活性。

YOLOv2 的目标不是简单地把网络做得更大，而是提高定位能力和 Recall、保持实时检测速度、提高特征提取网络效率，并让同一组模型权重能够适应不同输入分辨率。

### 1.2 YOLOv2 的主要改进

| 改进 | 具体做法 | 主要解决的问题 |
|---|---|---|
| Batch Normalization | 在卷积层中加入 BN | 稳定训练、加速收敛、降低过拟合 |
| High Resolution Classifier | 分类预训练阶段增加 448×448 微调 | 缩小分类预训练与检测输入之间的分辨率差异 |
| Fully Convolutional + Anchor Boxes | 移除全连接检测头，使用卷积预测 Anchor 偏移 | 增加候选框数量，改善 Recall |
| Dimension Clusters | 对 GT 框宽高进行 K-means 聚类 | 让 Anchor 尺寸更贴合数据集 |
| Direct Location Prediction | 使用 Sigmoid 限制中心点偏移 | 提高 Anchor 回归训练稳定性 |
| Fine-Grained Features | 将高分辨率特征通过 Passthrough 融合到深层特征 | 补充小目标的细节信息 |
| Multi-Scale Training | 每隔一定迭代改变输入尺寸 | 让同一模型适应不同分辨率 |
| Darknet-19 | 设计更高效的 Backbone | 降低计算量，同时提升分类特征质量 |

这些改进并不是全部由 YOLO 作者首次提出。例如 Batch Normalization、Anchor Boxes、Global Average Pooling 等均有已有研究基础。YOLOv2 的贡献在于将这些方法结合到统一的实时检测器中，并提出适合 YOLO 的 Anchor 聚类和位置参数化方式。

### 1.3 从 YOLOv1 到 YOLOv2 的整体变化

~~~mermaid
flowchart TD
    A["YOLOv1"] --> B["固定 7×7 Grid"]
    B --> C["每个 Grid 预测 2 个 Box"]
    C --> D["全连接层输出 7×7×30"]
    D --> E["YOLOv2"]
    E --> F["卷积检测头 + Anchor Boxes"]
    E --> G["Darknet-19 Backbone"]
    E --> H["Passthrough 特征融合"]
    E --> I["Multi-Scale Training"]
    F --> J["每个 Grid 预测 5 个 Anchor"]
    J --> K["13×13×(5×(5+C))"]
~~~

最重要的变化是：YOLOv2 不再直接预测固定数量的绝对 Bounding Box，而是以 Anchor 为先验，在每个网格位置预测多个 Anchor 的偏移量、Objectness 和类别概率。

---

## 2. 任务定义与输入输出表示

### 2.1 输入图片与特征图

YOLOv2 是 Fully Convolutional Detector，不再依赖 YOLOv1 最后的全连接层，因此输入空间尺寸可以变化。

以 $416\times416$ 为例：

$$
X\in\mathbb{R}^{B\times3\times416\times416}
$$

网络总下采样倍数为 32：

$$
416/32=13
$$

最终检测特征图为：

$$
F\in\mathbb{R}^{B\times1024\times13\times13}
$$

因此检测网格为：

$$
S_w=S_h=13
$$

每个网格位置对应原图中大约 $32\times32$ 像素的区域。

### 2.2 每个网格预测什么？

YOLOv2 在每个网格位置设置 5 个 Anchor Boxes。

每个 Anchor 预测：

$$
(t_x,t_y,t_w,t_h,t_o,p_1,\ldots,p_C)
$$

| 输出 | 含义 |
|---|---|
| $t_x,t_y$ | Bounding Box 中心位置的原始偏移量 |
| $t_w,t_h$ | Bounding Box 宽度、高度相对于 Anchor 的对数尺度偏移 |
| $t_o$ | Objectness 的原始输出 |
| $p_1,\ldots,p_C$ | 当前 Anchor 对应的类别条件概率 |

因此，每个 Anchor 输出 $5+C$ 个数值，每个网格输出：

$$
A(5+C)
$$

其中 $A=5$ 是 Anchor 数量。

### 2.3 输出维度计算

**PASCAL VOC：20 类**

$$
5\times(5+20)=125
$$

输出：

$$
Y\in\mathbb{R}^{B\times125\times13\times13}
$$

每张图片包含：

$$
13\times13\times5=845
$$

个候选 Bounding Box。

**COCO：80 类**

$$
5\times(5+80)=425
$$

输出：

$$
Y\in\mathbb{R}^{B\times425\times13\times13}
$$

每张图片仍然包含 845 个候选框，但每个候选框有 80 个类别分数。

**不同输入尺寸**

| 输入尺寸 | 输出 Grid | COCO 输出 Tensor | 候选框数量 |
|---:|---:|---|---:|
| 320×320 | 10×10 | $[B,425,10,10]$ | 500 |
| 416×416 | 13×13 | $[B,425,13,13]$ | 845 |
| 480×480 | 15×15 | $[B,425,15,15]$ | 1125 |
| 544×544 | 17×17 | $[B,425,17,17]$ | 1445 |
| 608×608 | 19×19 | $[B,425,19,19]$ | 1805 |

这些尺寸均为 32 的倍数。

这里用的是逻辑上的 Anchor 组织方式。Darknet 的实际输出按照 Anchor、坐标/类别字段和空间位置进行分段存储。可以将其理解为逻辑 Tensor：

$$
[B,H_g,W_g,A,5+C]
$$

但不要假设这个逻辑排列与 Darknet 原始内存排列完全相同。

### 2.4 与 YOLOv1 输出的区别

| 对比 | YOLOv1 | YOLOv2 |
|---|---|---|
| Grid | 7×7 | 13×13（416 输入） |
| 每个 Grid 的 Box 数量 | 2 | 5 个 Anchor |
| Box 参数 | $x,y,w,h,Confidence$ | $t_x,t_y,t_w,t_h,t_o$ |
| 类别预测 | 每个 Grid 一组 | 每个 Anchor 一组 |
| Box 尺寸先验 | 无 Anchor | K-means Anchor |
| 检测头 | 全连接层 | 卷积层 |
| 输出维度（VOC） | 7×7×30 | 13×13×125 |
| 候选框数量 | 98 | 845 |

YOLOv2 的类别预测从 Grid 级别变成了 Anchor 级别。同一网格内的不同 Anchor 可以分别预测不同类别，而不再被迫共享一组类别概率。

---

## 3. Anchor Boxes 与 Dimension Clusters

这是 YOLOv2 相比 YOLOv1 最重要的结构变化之一。

### 3.1 什么是 Anchor Box？

Anchor Box 可以理解为预先定义的一组 Bounding Box 尺寸先验。

假设某个网格位置有 5 个 Anchor：

$$
A_1=(p_{w1},p_{h1})
$$

$$
A_2=(p_{w2},p_{h2})
$$

$$
\cdots
$$

$$
A_5=(p_{w5},p_{h5})
$$

它们具有不同的宽高比例和尺寸。

网络不再从零开始预测任意大小的 Bounding Box，而是在 Anchor 的基础上预测偏移量。Anchor 提供一个合理的初始形状，让网络主要学习如何调整它。

### 3.2 为什么不用手工指定 Anchor？

Faster R-CNN 等 Anchor-based 方法通常会人工设置 Anchor 的尺寸和宽高比。但不同数据集的目标尺寸分布不同，人工指定可能不合适。

YOLOv2 使用 K-means 对训练集 GT Bounding Box 的宽度和高度进行聚类，让 Anchor 自动适应数据分布。

### 3.3 为什么 K-means 不使用普通欧氏距离？

如果使用欧氏距离：

$$
d=\sqrt{(w_1-w_2)^2+(h_1-h_2)^2}
$$

大框之间的尺寸差异更容易产生较大的距离值。但目标检测更关心 Bounding Box 的重叠质量，也就是 IoU。

因此 YOLOv2 使用：

$$
\boxed{
d(box,centroid)=1-IoU(box,centroid)
}
$$

这里比较的是框的宽高形状，通常将两个框的中心对齐，避免位置差异干扰尺寸聚类。IoU 越大，距离越小。

### 3.4 K-means 的执行过程

假设训练集有 $N$ 个 GT 框：

$$
G=\{(w_i,h_i)\}_{i=1}^{N}
$$

希望得到 $K=5$ 个 Anchor：

$$
A=\{(p_{wj},p_{hj})\}_{j=1}^{5}
$$

步骤：

1. 从 GT 框中初始化 5 个聚类中心。
2. 计算每个 GT 与 5 个 Anchor 的距离 $1-IoU$。
3. 将 GT 分配给距离最小的 Anchor。
4. 根据每个簇中的 GT 更新中心宽高。
5. 重复分配和更新，直到聚类中心稳定。

~~~mermaid
flowchart TD
    A["训练集 GT Bounding Boxes"] --> B["提取每个框的 w、h"]
    B --> C["初始化 K=5 个聚类中心"]
    C --> D["计算 1-IoU 距离"]
    D --> E["将 GT 分配给最近 Anchor"]
    E --> F["更新每个簇的宽高中心"]
    F --> G{"中心是否稳定？"}
    G -->|否| D
    G -->|是| H["得到 5 个 Anchor Priors"]
~~~

论文在 VOC 2007 上的结果：

| Anchor 生成方法 | Anchor 数量 | Average IoU |
|---|---:|---:|
| Cluster SSE | 5 | 58.7 |
| Cluster IoU | 5 | 61.0 |
| 手工 Anchor Boxes | 9 | 60.9 |
| Cluster IoU | 9 | 67.2 |

5 个基于 IoU 聚类的 Anchor 已经达到与 9 个手工 Anchor 接近的平均 IoU。作者选择 5 个 Anchor，是模型复杂度和 Recall 之间的折中。

### 3.5 官方配置中的 Anchor 数值

作者 Darknet 的 COCO 配置 **cfg/yolov2.cfg**：

~~~text
anchors =
0.57273, 0.677385,
1.87446, 2.06253,
3.33843, 5.47434,
7.88282, 3.52778,
9.77052, 9.16828
~~~

| Anchor | 宽度 | 高度 | 宽高比 |
|---|---:|---:|---|
| A1 | 0.57273 | 0.677385 | 偏小、接近正方形 |
| A2 | 1.87446 | 2.06253 | 中小、接近正方形 |
| A3 | 3.33843 | 5.47434 | 纵向较长 |
| A4 | 7.88282 | 3.52778 | 横向较宽 |
| A5 | 9.77052 | 9.16828 | 大尺寸、接近正方形 |

这些数值以 Grid Cell 为单位，而不是直接以像素为单位。

416 输入时，Stride=32。例如 A1 的像素尺寸约为：

$$
w=0.57273\times32\approx18.33
$$

$$
h=0.677385\times32\approx21.68
$$

A5 的像素尺寸约为：

$$
w=9.77052\times32\approx312.66
$$

$$
h=9.16828\times32\approx293.38
$$

VOC 配置中的 Anchor 数值不同：

~~~text
1.3221, 1.73145,
3.19275, 4.00944,
5.05587, 8.09892,
9.47112, 4.84053,
11.2364, 10.0071
~~~

这说明 Anchor 是与训练数据集相关的先验，不应把 COCO Anchor 直接认为是所有数据集的通用最优设置。

---

## 4. Direct Location Prediction：为什么要限制中心点偏移？

引入 Anchor 后，YOLOv2 还需要解决一个训练稳定性问题。

### 4.1 直接预测 Anchor 偏移的问题

在一些 Region Proposal Network 的参数化方式中，中心坐标可能表示为：

$$
x=t_xp_w+x_a
$$

$$
y=t_yp_h+y_a
$$

其中 $x_a,y_a$ 是 Anchor 中心位置，$p_w,p_h$ 是 Anchor 宽高，$t_x,t_y$ 是网络预测的偏移量。

这种参数化允许预测中心点偏离当前 Grid Cell 很远。在随机初始化阶段，网络可能输出较大的偏移值，使预测框跑到很远的位置，导致训练不稳定。

### 4.2 YOLOv2 的位置参数化

YOLOv2 使用 Logistic / Sigmoid 限制中心点偏移：

$$
\boxed{
b_x=\sigma(t_x)+c_x
}
$$

$$
\boxed{
b_y=\sigma(t_y)+c_y
}
$$

其中 $t_x,t_y$ 是网络原始输出，$\sigma$ 是 Sigmoid 函数，$c_x,c_y$ 是当前 Grid Cell 左上角对应的整数坐标。

$$
\sigma(t)=\frac{1}{1+e^{-t}}
$$

因此：

$$
0<\sigma(t)<1
$$

预测中心点被限制在当前 Grid Cell 内。

### 4.3 宽度和高度的参数化

宽度、高度以 Anchor 尺寸为基准，通过指数函数调整：

$$
\boxed{
b_w=p_we^{t_w}
}
$$

$$
\boxed{
b_h=p_he^{t_h}
}
$$

预测框宽高始终为正，网络可以在 Anchor 基础上放大或缩小。

### 4.4 Objectness 参数化

YOLOv2 的 Objectness 输出：

$$
\boxed{
\sigma(t_o)=P(Object)\times IoU(b,GT)
}
$$

论文将其解释为目标存在性与定位质量的综合分数。

---

## 5. 数值例子：从网络输出还原 Bounding Box

假设输入图片 416×416，Grid 为 13×13，Stride=32。某个目标中心点位于归一化坐标 $(0.35,0.60)$，GT 宽度为 64 px，高度为 80 px。

### 5.1 确定 Grid Cell

$$
i=\lfloor0.35\times13\rfloor=4
$$

$$
j=\lfloor0.60\times13\rfloor=7
$$

所以目标中心位于第 5 列、第 8 行（从 1 开始计数）。

### 5.2 计算 Grid 内偏移量

$$
x_{offset}=0.35\times13-4=0.55
$$

$$
y_{offset}=0.60\times13-7=0.80
$$

因此：

$$
\sigma(t_x)=0.55,\quad \sigma(t_y)=0.80
$$

反过来：

$$
t_x=\ln\frac{0.55}{1-0.55}\approx0.2007
$$

$$
t_y=\ln\frac{0.80}{1-0.80}\approx1.3863
$$

### 5.3 将 GT 宽高转换为 Grid 单位

416 输入、13 个 Grid，每个 Grid 对应 32 px：

$$
w_{gt}=\frac{64}{32}=2
$$

$$
h_{gt}=\frac{80}{32}=2.5
$$

假设选择的 Anchor 为：

$$
p_w=1.87446,\quad p_h=2.06253
$$

计算宽高偏移：

$$
t_w=\ln\frac{2}{1.87446}\approx0.06483
$$

$$
t_h=\ln\frac{2.5}{2.06253}\approx0.19236
$$

### 5.4 还原预测框

$$
b_x=4+0.55=4.55
$$

$$
b_y=7+0.80=7.80
$$

$$
b_w=1.87446e^{0.06483}\approx2
$$

$$
b_h=2.06253e^{0.19236}\approx2.5
$$

转换到整张图片的归一化坐标：

$$
x_c=\frac{4.55}{13}=0.35,\quad
y_c=\frac{7.80}{13}=0.60
$$

$$
w=\frac{2}{13}\approx0.15385,\quad
h=\frac{2.5}{13}\approx0.19231
$$

再转换为像素：

$$
x_c=0.35\times416=145.6
$$

$$
y_c=0.60\times416=249.6
$$

$$
w=64,\quad h=80
$$

最终得到：

$$
Box=(145.6,249.6,64,80)
$$

这里的坐标形式为中心点、宽度和高度。

---

## 6. 完整网络架构（Architecture Deep Dive）

YOLOv2 的 Backbone 是 Darknet-19。

### 6.1 Darknet-19 的设计思想

Darknet-19 是作者为 YOLOv2 设计的分类网络，主要特点：

- 19 个卷积层。
- 5 个 MaxPooling 层。
- 主要使用 3×3 卷积。
- 使用 1×1 卷积压缩通道。
- 卷积层后使用 Batch Normalization。
- 使用 Leaky ReLU。
- 分类网络末端使用 Global Average Pooling 和 Softmax。

论文报告 Darknet-19 在 ImageNet 上：

| 指标 | 结果 |
|---|---:|
| Top-1 Accuracy | 72.9% |
| Top-5 Accuracy | 91.2% |
| 计算量 | 5.58 Billion Operations |

这里的计算量对应论文中的分类网络输入设置，不应直接当作 YOLOv2 检测模型在任意分辨率下的 FLOPs。

### 6.2 为什么使用 1×1 卷积？

例如：

$$
[B,128,H,W]\xrightarrow{1\times1}[B,64,H,W]
$$

再经过：

$$
[B,64,H,W]\xrightarrow{3\times3}[B,128,H,W]
$$

1×1 卷积先将通道数从 128 压缩到 64，减少后续 3×3 卷积的计算量。

卷积计算量近似为：

$$
FLOPs\propto H\times W\times C_{in}\times C_{out}\times K^2
$$

### 6.3 Batch Normalization

卷积层输出：

$$
z=Wx+b
$$

Batch Normalization 对 Batch 内的激活进行归一化：

$$
\hat z=\frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

然后进行可学习的缩放和平移：

$$
y=\gamma\hat z+\beta
$$

YOLOv2 在卷积层中加入 BN，论文报告带来超过 2% 的 mAP 提升，同时帮助模型稳定训练。加入 BN 后，作者可以移除 Dropout，而不会出现明显过拟合。

### 6.4 Leaky ReLU

Darknet-19 使用 Leaky ReLU：

$$
f(x)=
\begin{cases}
x,&x>0\\
0.1x,&x\leq0
\end{cases}
$$

### 6.5 Darknet-19 分类网络结构

以 224×224 输入为例，论文 Table 6 的主要结构：

| 阶段 | 运算 | 输出尺寸 | 通道 |
|---|---|---:|---:|
| 1 | Conv 3×3 + MaxPool | 112×112 | 32 |
| 2 | Conv 3×3 + MaxPool | 56×56 | 64 |
| 3 | Conv 3×3 → Conv 1×1 → Conv 3×3 + MaxPool | 28×28 | 128 |
| 4 | Conv 3×3 → Conv 1×1 → Conv 3×3 + MaxPool | 14×14 | 256 |
| 5 | Conv 3×3 → Conv 1×1 → Conv 3×3 → Conv 1×1 → Conv 3×3 + MaxPool | 7×7 | 512 |
| 6 | Conv 3×3 → Conv 1×1 → Conv 3×3 → Conv 1×1 → Conv 3×3 | 7×7 | 1024 |
| Classifier | Conv 1×1 → Global Average Pooling → Softmax | 1×1 | 1000 |

这里描述的是 Darknet-19 分类网络。检测网络会移除分类输出层，并在其基础上增加检测头。

### 6.6 YOLOv2 检测网络的整体结构

YOLOv2 将 Darknet-19 改造成检测网络：

1. 移除分类网络最后的 1000 类输出卷积。
2. 增加 3 个 3×3、1024 通道的卷积层。
3. 增加一个 1×1 检测输出卷积。
4. 从较早的 26×26 特征图引出一条 Passthrough 分支。
5. 将 Passthrough 特征与深层特征融合。

~~~mermaid
flowchart TD
    A["Input 416×416×3"] --> B["Darknet-19 Backbone"]
    B --> C["Deep Feature 13×13×1024"]
    B --> D["Shallow Feature 26×26×512"]
    C --> E["3×3 Conv 1024"]
    E --> F["Feature Fusion"]
    D --> G["1×1 Conv 64"]
    G --> H["Reorg / Space-to-Depth"]
    H --> I["13×13×256"]
    I --> F
    F --> J["3×3 Conv 1024"]
    J --> K["1×1 Conv: 425"]
    K --> L["Region Layer"]
    L --> M["13×13×5 Anchors"]
~~~

上图是便于理解的数据流示意。实际层间连接以 Darknet 配置文件中的 route、reorg 和卷积层为准。

### 6.7 为什么检测网络还需要额外卷积层？

分类任务最终需要区分图像属于哪一个类别；检测任务还需要输出每个位置的框、Objectness 和类别信息。

因此 YOLOv2 在分类 Backbone 的深层特征上继续增加卷积层，让特征适合目标定位与检测预测。

---

## 7. Passthrough Layer：细粒度特征融合

### 7.1 为什么需要 Passthrough？

Darknet-19 经过多次下采样后，最终输出特征图只有 13×13。深层特征具有较强的语义信息，但空间分辨率较低。

对于小目标，经过多次下采样后，一些局部细节可能已经丢失。因此 YOLOv2 从较早的 26×26 特征图中提取细粒度信息，再与 13×13 的深层特征融合。

### 7.2 Passthrough 的输入

416×416 输入时：

- 深层特征：13×13×1024。
- 浅层特征：26×26×512。

浅层特征通道数为 512，先通过 1×1 卷积压缩：

$$
26\times26\times512
\rightarrow
26\times26\times64
$$

然后使用 Reorg，Stride=2。

### 7.3 Reorg / Space-to-Depth

Reorg 不执行插值，也不使用卷积核学习参数，而是重新排列特征元素。

输入：

$$
X\in\mathbb{R}^{B\times64\times26\times26}
$$

Stride=2 后：

$$
Y\in\mathbb{R}^{B\times256\times13\times13}
$$

维度变化：

$$
H'=H/2,\quad W'=W/2,\quad C'=C\times2^2=4C
$$

也就是：

$$
26\times26\times64
\rightarrow
13\times13\times256
$$

原来每个 2×2 空间邻域中的 4 个位置，被重新排列到 4 组通道中。空间分辨率减半，通道数增加四倍，但没有通过池化直接丢弃这 2×2 区域中的特征值。

### 7.4 与深层特征拼接

浅层分支经过 Reorg：

$$
13\times13\times256
$$

深层分支：

$$
13\times13\times1024
$$

在通道维度 Concat：

$$
13\times13\times(256+1024)
=\boxed{13\times13\times1280}
$$

随后经过 3×3 卷积：

$$
13\times13\times1280
\rightarrow
13\times13\times1024
$$

最后通过 1×1 卷积得到预测结果。

### 7.5 Passthrough 与 FPN 的区别

| 对比 | YOLOv2 Passthrough | FPN |
|---|---|---|
| 特征融合 | Reorg 后 Concat | 上采样与横向连接 |
| 空间处理 | Space-to-Depth | Upsampling |
| 通道融合 | Concat 后卷积 | Add 或 Concat 后卷积 |
| 检测输出 | 单个融合后的输出尺度 | 通常多个尺度输出 |
| 核心目的 | 将高分辨率细节注入深层特征 | 构建多尺度语义特征金字塔 |

YOLOv2 不是现代意义上具有 P3、P4、P5 多个独立检测头的 FPN/PAN 检测器。它把高分辨率特征融合到单个最终检测尺度中。

论文消融实验中，Passthrough 带来约 1% 的 VOC 2007 mAP 提升。但它并不能完全解决小目标检测问题，因为最终仍然只有一个检测输出尺度。

---

## 8. 单张图片完整前向传播（Forward Pass）

以 416×416 输入、COCO 80 类为例，追踪一张图片经过 YOLOv2 的完整数据流。

### 8.1 输入

$$
X\in\mathbb{R}^{1\times3\times416\times416}
$$

### 8.2 Backbone 特征提取

| 阶段 | 操作 | 输出 Shape（NCHW） |
|---|---|---|
| Input | RGB 图片 | $[1,3,416,416]$ |
| 1 | Conv 3×3, 32 | $[1,32,416,416]$ |
| 2 | MaxPool 2×2, 2 | $[1,32,208,208]$ |
| 3 | Conv 3×3, 64 | $[1,64,208,208]$ |
| 4 | MaxPool 2×2, 2 | $[1,64,104,104]$ |
| 5 | Conv 128 → 64 → 128 | $[1,128,104,104]$ |
| 6 | MaxPool 2×2, 2 | $[1,128,52,52]$ |
| 7 | Conv 256 → 128 → 256 | $[1,256,52,52]$ |
| 8 | MaxPool 2×2, 2 | $[1,256,26,26]$ |
| 9 | Conv 512 → 256 → 512 → 256 → 512 | $[1,512,26,26]$ |
| 10 | MaxPool 2×2, 2 | $[1,512,13,13]$ |
| 11 | Conv 1024 → 512 → 1024 → 512 → 1024 | $[1,1024,13,13]$ |

其中第 9 阶段的 26×26×512 特征会被引出，作为 Passthrough 分支输入。

### 8.3 检测头主分支

在深层特征上增加两个 3×3 卷积：

$$
[1,1024,13,13]\rightarrow[1,1024,13,13]
$$

### 8.4 Passthrough 分支

浅层特征：

$$
[1,512,26,26]
$$

1×1 卷积：

$$
[1,512,26,26]\rightarrow[1,64,26,26]
$$

Reorg：

$$
[1,64,26,26]\rightarrow[1,256,13,13]
$$

### 8.5 Concat 与检测输出

深层特征和浅层特征 Concat：

$$
[1,1024+256,13,13]
=\boxed{[1,1280,13,13]}
$$

3×3 卷积：

$$
[1,1280,13,13]\rightarrow[1,1024,13,13]
$$

1×1 卷积：

$$
[1,1024,13,13]\rightarrow[1,425,13,13]
$$

其中：

$$
425=5\times(4+1+80)
$$

最终得到 845 个 Anchor，每个 Anchor 有 85 个预测值。

~~~mermaid
flowchart TD
    A["Input [1,3,416,416]"] --> B["Darknet-19"]
    B --> C["Deep [1,1024,13,13]"]
    B --> D["Shallow [1,512,26,26]"]
    C --> E["Extra Conv 1024"]
    D --> F["Conv 1×1: 64"]
    F --> G["Reorg: [1,256,13,13]"]
    E --> H["Concat"]
    G --> H
    H --> I["[1,1280,13,13]"]
    I --> J["Conv 3×3: 1024"]
    J --> K["Conv 1×1: 425"]
    K --> L["Region Decode"]
    L --> M["845 Boxes + Class Scores"]
~~~

### 8.6 如果输入改成 608×608

由于总下采样倍数为 32：

$$
608/32=19
$$

最终特征图变成：

$$
[1,1024,19,19]
$$

Passthrough 分支：

$$
38\times38\times512
\rightarrow
38\times38\times64
\rightarrow
19\times19\times256
$$

融合后：

$$
19\times19\times1280
$$

最终输出：

$$
[1,425,19,19]
$$

模型结构和权重不需要因为输入尺寸从 416 变为 608 而改变，只需要特征图空间尺寸随输入变化。

---

# 阶段 B：理解训练与推理（Understand Training & Inference）

## 9. 参数初始化与预训练

### 9.1 Darknet-19 ImageNet 预训练

论文先在 ImageNet 1000 类分类任务上训练 Darknet-19：

| 参数 | 设置 |
|---|---|
| Epoch | 160 |
| Optimizer | SGD |
| 初始 Learning Rate | 0.1 |
| Learning Rate Scheduler | Polynomial Decay，Power=4 |
| Weight Decay | 0.0005 |
| Momentum | 0.9 |
| 输入分辨率 | 224×224 |

训练过程中使用 Random Crop、Rotation、Hue Shift、Saturation Shift 和 Exposure Shift。

### 9.2 High Resolution Classifier

YOLOv1 的分类预训练使用 224×224 输入，而检测时使用 448×448。分辨率变化较大，导致网络需要同时适应检测任务和输入分辨率变化。

YOLOv2 先在 224×224 上训练 Darknet-19，再将分类网络提高到 448×448，在 ImageNet 上微调 10 Epoch，最后将 Backbone 转换为检测网络并进行检测训练。

448×448 微调后，Darknet-19 的 ImageNet 指标达到：

| 指标 | 结果 |
|---|---:|
| Top-1 | 76.5% |
| Top-5 | 93.3% |

论文报告 High Resolution Classifier 带来接近 4% 的检测 mAP 提升。

### 9.3 检测网络初始化

检测网络保留 Darknet-19 的卷积特征提取层，移除原来的分类输出层，并增加检测头。

因此 Backbone 使用 ImageNet 预训练权重，新增加的检测卷积层需要初始化，最终预测层需要适配检测类别数和 Anchor 数量。

Darknet 历史训练中常见的预训练卷积权重文件为 **darknet19_448.conv.23**。该文件用于初始化 YOLOv2 的卷积特征提取部分，而不是完整的检测模型权重。

---

## 10. 正负样本分配（Anchor Assignment）

YOLOv2 的正负样本分配与 YOLOv1 有明显区别。

### 10.1 YOLOv1 的 Responsible Predictor

YOLOv1 对于一个包含 GT 的网格：

1. 计算该网格两个预测框与 GT 的 IoU。
2. IoU 最大的预测框成为 Responsible Predictor。
3. 只有它计算坐标损失。
4. 其他预测框计算 No-Object Confidence Loss。

也就是说，YOLOv1 是根据当前预测框与 GT 的 IoU 选择负责预测器。

### 10.2 YOLOv2 的 Anchor Assignment

YOLOv2 不再直接使用当前预测框的 IoU 来选择正样本 Anchor。

对于一个 GT：

1. 根据 GT 中心点找到所在 Grid Cell。
2. 取出该 Grid Cell 的 5 个 Anchor。
3. 将 GT 与 Anchor 的中心对齐。
4. 只比较 Anchor 与 GT 的宽高形状。
5. 计算 IoU。
6. 选择 IoU 最大的 Anchor 负责这个 GT。

数学表示：

$$
j^*=\arg\max_j IoU(A_j,GT_{shape})
$$

其中 $A_j$ 是第 j 个 Anchor 的尺寸，$GT_{shape}$ 是将 GT 中心移动到原点后得到的宽高框。

因此，YOLOv2 的正样本分配主要依赖 Anchor 先验与 GT 尺寸之间的匹配，而不是当前网络预测框的位置质量。

### 10.3 为什么匹配时忽略中心点？

Anchor 是尺寸先验，它本身并不携带具体位置。一个 GT 的中心点已经决定了它属于哪个 Grid Cell。

因此在选择 Anchor 时，只需要比较宽高：

$$
IoU(A_j,GT_{shape})
$$

不需要再比较中心点位置。

Darknet 配置中的 **bias_match=1** 对应这种 Anchor 尺寸匹配行为。

### 10.4 正负样本分配示例

假设当前 GT 的中心位于 Grid Cell：

$$
(i,j)=(4,7)
$$

当前 GT 的尺寸为：

$$
(w,h)=(2.0,2.5)
$$

5 个 Anchor 与 GT 的尺寸 IoU：

| Anchor | IoU |
|---|---:|
| A1 | 0.08 |
| A2 | 0.72 |
| A3 | 0.31 |
| A4 | 0.19 |
| A5 | 0.12 |

因此：

$$
j^*=2
$$

A2 被选为负责预测该 GT 的正样本 Anchor。其余 Anchor 不负责这个 GT。

### 10.5 No-Object Ignore 机制

Darknet 的 Region Layer 还有一个重要逻辑：

- 默认情况下，未负责 GT 的 Anchor 会受到 No-Object Confidence 约束。
- 如果某个预测框与任意 GT 的 IoU 已经超过阈值，则不再把它当作普通背景框惩罚。

官方 **cfg/yolov2.cfg** 中：

$$
thresh=0.6
$$

源码逻辑可以概括为：

~~~python
for each_grid_cell:
    for each_anchor:
        best_iou = max_iou(predicted_box, all_ground_truth_boxes)

        # 默认作为背景预测
        apply_no_object_confidence_loss(target=0)

        # 与某个 GT 已经高度重叠，不再惩罚
        if best_iou > ignore_threshold:
            ignore_no_object_loss()

for each_ground_truth:
    cell = grid_cell_containing_gt_center
    anchor = best_shape_iou_anchor(cell, gt)

    assign_positive(anchor, gt)
~~~

注意：**thresh=0.6** 是 Region Layer 训练中用于 No-Object Ignore 的阈值，不是推理时的类别置信度阈值，也不是 NMS IoU 阈值。

### 10.6 与 YOLOv1 的正样本分配对比

| 对比 | YOLOv1 | YOLOv2 |
|---|---|---|
| 每个网格的预测器 | 2 个自由预测框 | 5 个 Anchor |
| 选择负责预测器 | 当前预测框与 GT 的 IoU | Anchor 尺寸与 GT 尺寸的 IoU |
| 匹配时是否比较中心位置 | 预测框本身包含位置 | 先固定 GT 所在 Grid，匹配时比较形状 |
| 负责预测器数量 | 一个 GT 由一个预测框负责 | 一个 GT 由一个 Anchor 负责 |
| 背景预测处理 | No-Object Confidence Loss | No-Object Loss + Ignore Threshold |

---

## 11. YOLOv2 Loss Function

论文说明了 YOLOv2 的预测参数和训练目标，但没有像现代一些论文一样给出一份完整展开的统一 Loss 公式。下面先根据论文定义训练目标，再结合 Darknet **src/region_layer.c** 解释实际梯度信号。

### 11.1 Bounding Box 坐标目标

网络输出原始值：

$$
(t_x,t_y,t_w,t_h)
$$

GT 的中心点位于 Grid Cell $(i,j)$，归一化中心坐标为 $(x,y)$，归一化宽高为 $(w,h)$。

目标值：

$$
t_x^*=xS_w-i
$$

$$
t_y^*=yS_h-j
$$

$$
t_w^*=\ln\frac{wS_w}{p_w}
$$

$$
t_h^*=\ln\frac{hS_h}{p_h}
$$

其中 $S_w,S_h$ 是 Grid 的宽、高，$p_w,p_h$ 是 Anchor 宽、高，单位为 Grid Cell，$i,j$ 是 GT 中心所在 Grid Cell 的索引。

坐标回归需要让网络输出接近这些目标值。

### 11.2 Coordinate Loss 的权重

Darknet 源码中，坐标梯度使用：

$$
scale=\lambda_{coord}(2-wh)
$$

其中 $w,h$ 是相对于整张图片归一化的 GT 宽高。

因此：

$$
L_{box}\propto
\sum_{positive}
\lambda_{coord}(2-wh)
\left[
(t_x-t_x^*)^2+
(t_y-t_y^*)^2+
(t_w-t_w^*)^2+
(t_h-t_h^*)^2
\right]
$$

这里是根据源码坐标误差信号整理出的教学表达。论文没有逐字给出这一完整公式，具体实现应以 **delta_region_box()** 中的目标值和梯度计算为准。

小目标的 $wh$ 较小，权重接近 2；大目标的 $wh$ 较大，权重降低。这让小目标的坐标误差在训练中获得更高的相对权重。这与 YOLOv1 使用平方根宽高误差的方式不同。

### 11.3 Objectness Loss

YOLOv2 的 Objectness 预测目标：

$$
Objectness^*=P(Object)\times IoU
$$

对于负责 GT 的正样本 Anchor：

$$
Objectness^*=IoU(pred,GT)
$$

当官方配置 **rescore=1** 时，Darknet 使用：

$$
\Delta_o\propto IoU-\sigma(t_o)
$$

可以将其理解为让预测 Objectness 接近当前预测框与 GT 的 IoU。

如果 **rescore=0**，正样本目标则是 1。作者仓库中的标准 **yolov2.cfg** 使用 **rescore=1**，而 **yolov2-tiny.cfg** 使用 **rescore=0**。这是一个需要注意的配置差异。

### 11.4 No-Object Loss

对普通负样本 Anchor：

$$
Objectness^*=0
$$

因此：

$$
\Delta_{noobj}\propto0-\sigma(t_o)
$$

其权重由 **noobject_scale** 控制。

但如果预测框与某个 GT 的 IoU 已经超过 **thresh**，源码会跳过该 Anchor 的 No-Object 惩罚。

### 11.5 Classification Loss

YOLOv2 每个 Anchor 都有自己的类别预测向量。

对于一个正样本 Anchor，类别标签为 One-Hot：

$$
y_c=
\begin{cases}
1,&c=c_{gt}\\
0,&c\neq c_{gt}
\end{cases}
$$

标准 YOLOv2 配置使用 Softmax：

$$
P(c\mid Object)=Softmax(z_c)
$$

Darknet 源码为类别分支构造的梯度信号是：

$$
\Delta_c\propto y_c-P(c\mid Object)
$$

这对应 Softmax 分类的交叉熵梯度形式。背景 Anchor 不需要计算类别损失。

### 11.6 完整训练目标的组成

可以将 YOLOv2 的 Region Layer 训练目标归纳为：

$$
L=
\lambda_{coord}L_{box}
+\lambda_{obj}L_{obj}
+\lambda_{noobj}L_{noobj}
+\lambda_{class}L_{class}
$$

| Loss | 参与样本 | 监督目标 |
|---|---|---|
| $L_{box}$ | 正样本 Anchor | $t_x^*,t_y^*,t_w^*,t_h^*$ |
| $L_{obj}$ | 正样本 Anchor | IoU（rescore=1） |
| $L_{noobj}$ | 负样本 Anchor | Objectness=0 |
| $L_{class}$ | 正样本 Anchor | 类别 One-Hot |

标准 Darknet COCO / VOC 配置中的权重：

| 配置项 | 数值 |
|---|---:|
| coord_scale | 1 |
| object_scale | 5 |
| noobject_scale | 1 |
| class_scale | 1 |
| rescore | 1 |
| thresh | 0.6 |

注意：这些是 Darknet 配置的权重和阈值，不应把它们说成论文中统一给出的所有数据集通用超参数。

### 11.7 Warmup：为什么训练开始时要先稳定 Anchor？

Darknet **region_layer.c** 中还有一个实现细节：

当：

$$
net.seen<12800
$$

源码会对每个 Grid Cell 的每个 Anchor 施加一个较小的坐标更新，让预测框先接近当前 Grid Cell 的中心点和当前 Anchor 的宽高。坐标更新尺度为 0.01。

这样做的目的是让模型在训练初期先学会使用 Anchor 的基本位置和尺寸，避免随机初始化的预测框偏离过大。

这属于 Darknet 参考实现中的 Warmup 行为，不能将它误写成论文单独提出的一项网络模块。

---

## 12. Multi-Scale Training

### 12.1 为什么需要多尺度训练？

YOLOv2 是全卷积网络，没有固定尺寸的全连接预测头。因此，只要输入尺寸与网络下采样倍数兼容，就可以改变输入分辨率。

不同输入尺寸会影响最终特征图大小、候选框数量、小目标的特征表达、推理速度和显存占用。

### 12.2 论文中的 Multi-Scale Training

论文采用的策略：

1. 每训练 10 个 Batch。
2. 随机选择新的输入尺寸。
3. 输入尺寸从 320 到 608。
4. 每次增加 32。
5. 调整网络输入尺寸并继续训练。

尺寸集合：

$$
\{320,352,384,416,448,480,512,544,576,608\}
$$

由于总下采样倍数为 32，所以这些尺寸都可以得到整数 Grid。

### 12.3 不同输入尺寸对推理的影响

输入 416：

$$
Grid=13\times13
$$

$$
13\times13\times5=845
$$

输入 608：

$$
Grid=19\times19
$$

$$
19\times19\times5=1805
$$

更高输入分辨率会增加候选框数量和计算量，也会给小目标保留更多空间信息。

### 12.4 速度与精度的取舍

| 输入尺寸 | VOC 2007 mAP | FPS |
|---:|---:|---:|
| 288×288 | 69.0 | 91 |
| 352×352 | 73.7 | 81 |
| 416×416 | 76.8 | 67 |
| 480×480 | 77.8 | 59 |
| 544×544 | 78.6 | 40 |

这些结果来自论文 Table 3，使用的是同一组训练权重，只在推理时改变输入尺寸。测试硬件为原版 NVIDIA GeForce GTX Titan X，不是 Pascal Titan X。

这意味着 YOLOv2 可以在不重新训练的情况下，通过改变输入尺寸在速度和精度之间进行权衡。

---

## 13. 推理过程与后处理（Inference Pipeline）

### 13.1 输入预处理

推理时将输入图片 Resize 到网络指定尺寸，并转换为网络要求的 RGB / Tensor 格式。Darknet 的检测框后处理还需要考虑 Resize / Letterbox 对坐标的影响。

### 13.2 网络前向传播

以 416×416、COCO 80 类为例：

$$
X\rightarrow Darknet19
\rightarrow Passthrough
\rightarrow Detection\ Head
$$

最终输出：

$$
Y\in\mathbb{R}^{B\times425\times13\times13}
$$

### 13.3 Bounding Box Decode

网络输出：

$$
(t_x,t_y,t_w,t_h,t_o)
$$

通过：

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y
$$

$$
b_w=p_we^{t_w}
$$

$$
b_h=p_he^{t_h}
$$

得到 Grid 坐标系中的 Bounding Box。再将中心坐标和宽高除以 Grid 尺寸，转换为整张图片归一化坐标。

### 13.4 类别分数计算

每个 Anchor 的 Objectness：

$$
O=\sigma(t_o)
$$

每个类别的条件概率：

$$
P(c\mid Object)=Softmax(z_c)
$$

最终类别分数：

$$
\boxed{
Score(c)=O\times P(c\mid Object)
}
$$

例如 Objectness=0.75、Cat 类别概率=0.80、Dog 类别概率=0.10：

$$
Score(Cat)=0.75\times0.80=0.60
$$

$$
Score(Dog)=0.75\times0.10=0.075
$$

### 13.5 与 YOLOv1 Confidence 的关系

YOLOv1 的 Confidence 定义：

$$
Confidence=P(Object)\times IoU
$$

YOLOv2 的 Objectness 定义：

$$
Objectness=P(Object)\times IoU
$$

因此，两者在语义上是连续的。

区别在于：YOLOv1 每个 Grid 预测两个 Confidence，类别概率由 Grid 共享；YOLOv2 每个 Anchor 独立预测 Objectness 和类别概率。

YOLOv2 并没有把 Objectness 定义成一个完全独立于定位质量的纯目标存在概率。它仍然包含 IoU 质量信息。

### 13.6 NMS

YOLOv2 仍然需要 NMS，不是 NMS-free 模型。

典型流程：

1. 解码每个 Anchor 的 Bounding Box。
2. 计算每个 Anchor 的类别分数。
3. 使用 Score Threshold 过滤低分候选。
4. 对每个类别分别执行 NMS。
5. 保留最终检测框。

例如同一类别有两个候选框，A 的 Score 为 0.90，B 的 Score 为 0.80。如果 $IoU(A,B)=0.85$ 且 NMS 阈值为 0.5，则保留 A，抑制 B。

### 13.7 训练与推理对比

| 项目 | Training | Inference |
|---|---|---|
| 输入 | 图片和 GT | 只有图片 |
| Forward | Backbone + Passthrough + Head | 同样的网络 |
| Anchor Assignment | 需要 | 不需要 |
| Box Loss | 计算 | 不计算 |
| Objectness Loss | 计算 | 不计算 |
| Classification Loss | 计算 | 不计算 |
| Backpropagation | 需要 | 不需要 |
| Box Decode | 用于匹配与计算 IoU | 用于输出框 |
| Score | 用于监督 | Objectness × Class Probability |
| NMS | 不需要 | 需要 |

---

## 14. YOLO9000：WordTree 与检测/分类联合训练

论文标题中的 Stronger 部分，不只是 YOLOv2 的网络结构改进，而是提出了一个更大类别词表的联合训练方法。

### 14.1 为什么需要 YOLO9000？

目标检测数据集的标注成本远高于图像分类。例如 COCO 提供 Bounding Box 和类别标签，而 ImageNet 提供大量图片级类别标签，但并不为每张图片提供完整 Bounding Box。

如果只使用检测数据训练，模型只能学习有限的检测类别。如果能利用 ImageNet 这样的分类数据，就可以扩大模型认识的类别范围。

问题在于：如何把不同数据集中的类别标签合并到一个统一的分类体系中？

### 14.2 为什么不能直接使用普通 Softmax？

假设有两个类别：Dog 和 Norfolk Terrier。Norfolk Terrier 是 Dog 的一种。

如果将它们放进同一个普通 Softmax 中，模型会认为两个类别互斥。但现实中：

$$
Norfolk\ Terrier\Rightarrow Dog
$$

它们存在上下位关系。COCO 可能只标注 Dog，而 ImageNet 可能标注 Norfolk Terrier。如果直接合并两个数据集，普通 Softmax 的互斥假设就会产生冲突。

### 14.3 WordNet 与 WordTree

WordNet 是一个描述词汇语义关系的数据库，其中包含大量概念之间的上下位关系。

~~~mermaid
flowchart TD
    A["Physical Object"] --> B["Animal"]
    B --> C["Mammal"]
    C --> D["Canine"]
    D --> E["Dog"]
    E --> F["Terrier"]
    F --> G["Norfolk Terrier"]
    F --> H["Yorkshire Terrier"]
    F --> I["Bedlington Terrier"]
~~~

WordNet 原始结构是一个有向图，某些类别可能有多个父节点。YOLO9000 将其简化为一棵树，构建 WordTree。

### 14.4 Hierarchical Classification

普通分类通过一个 Softmax 在所有类别之间计算概率：

$$
P(Class\mid Image)
$$

WordTree 则在每个父节点下，对它的子节点进行 Softmax。

例如 Terrier 节点：

$$
P(Norfolk\ Terrier\mid Terrier)
$$

$$
P(Yorkshire\ Terrier\mid Terrier)
$$

$$
P(Bedlington\ Terrier\mid Terrier)
$$

如果希望计算 Norfolk Terrier 的绝对概率，需要将从根节点到该类别路径上的条件概率相乘：

$$
\begin{aligned}
P(Norfolk\ Terrier)={}&
P(Animal)\\
&\times P(Mammal\mid Animal)\\
&\times P(Canine\mid Mammal)\\
&\times P(Dog\mid Canine)\\
&\times P(Terrier\mid Dog)\\
&\times P(Norfolk\ Terrier\mid Terrier)
\end{aligned}
$$

论文在分类场景中将根节点 Physical Object 的概率设为 1。

### 14.5 为什么 WordTree 能够合并不同数据集？

假设 COCO 标签为 Dog，ImageNet 标签为 Norfolk Terrier。这两个标签可以映射到 WordTree 的不同层级：

- Dog 是父节点。
- Norfolk Terrier 是更细粒度的子节点。

当训练样本只标注 Dog 时，模型只需要学习到 Dog 这一层，不需要判断它到底是 Norfolk Terrier 还是 Yorkshire Terrier。当样本明确标注 Norfolk Terrier 时，才继续向下学习细分类别。

### 14.6 YOLO9000 的 9418 个类别

论文将 COCO 检测类别和 ImageNet 分类类别映射到 WordTree，最终得到：

$$
C=9418
$$

个类别节点。这不是简单地将 9418 个类别塞进普通 Softmax，而是利用层级结构组织条件概率。

### 14.7 检测与分类联合训练

YOLO9000 使用两种数据：

1. 有 Bounding Box 的检测数据。
2. 只有 Image-level Label 的分类数据。

训练时混合两种数据，并根据数据类型计算不同的 Loss。

**检测数据：**

- 计算 Bounding Box Loss。
- 计算 Objectness Loss。
- 计算分类树上的 Classification Loss。
- 使用完整检测训练流程。

**分类数据：**

- 不存在完整 Bounding Box 标注。
- 不计算普通检测框坐标回归 Loss。
- 选择对该类别预测概率最高的候选框。
- 只对该类别对应的 WordTree 路径计算分类监督。

论文还描述了对候选框重叠情况作出至少 0.3 IoU 的假设，用于分类数据中的 Objectness 处理。这里不能把分类标签当作精确的 Bounding Box 标注。

### 14.8 数据集比例

ImageNet 分类数据远大于 COCO 检测数据。为了平衡两类数据，作者对 COCO 进行过采样，使 ImageNet 数据量约为 COCO 的 4 倍：

$$
ImageNet:COCO\approx4:1
$$

### 14.9 为什么 YOLO9000 只使用 3 个 Anchor？

YOLO9000 的类别数量达到 9418。如果仍然使用 5 个 Anchor，每个 Anchor 都需要预测 9418 个类别分数，输出通道数会非常大。

因此论文将 Anchor 数量减少到 3 个：

$$
3\times(5+9418)=\boxed{28269}
$$

作者官方 **cfg/yolo9000.cfg** 中也使用：

- classes=9418
- num=3
- filters=28269
- height=544
- width=544

### 14.10 YOLO9000 的实验结果

在 ImageNet Detection 验证任务中：

- 共有 200 个检测类别。
- 其中只有 44 个类别与 COCO 检测数据重叠。
- 其余 156 个类别没有对应的检测标注训练数据。

YOLO9000 的结果：

| 评估范围 | mAP |
|---|---:|
| ImageNet Detection 全部类别 | 19.7 |
| 没有检测标注训练数据的 156 类 | 16.0 |

这个结果说明：分类数据能够帮助模型学习类别语义，并将一部分分类知识迁移到检测任务中。

但模型并不是凭空学会了这些类别的精确定位能力。检测数据仍然为模型提供了定位和 Objectness 训练信号，分类数据主要扩展了类别知识。

---

# 阶段 C：评价模型（Evaluate the Model）

## 15. 数据集与实验设置

### 15.1 ImageNet 分类预训练

Darknet-19 使用 ImageNet 1000 类分类数据集进行预训练。训练分为 224×224 分类训练、448×448 高分辨率微调，以及将 Backbone 转换为检测网络。

### 15.2 PASCAL VOC

YOLOv2 在 VOC 2007 和 VOC 2012 上进行检测实验。VOC 有 20 个类别，因此每个 Anchor 输出：

$$
5+20=25
$$

5 个 Anchor 的输出通道数：

$$
5\times25=125
$$

### 15.3 COCO

COCO 检测任务使用 80 个类别。YOLOv2 的输出通道数：

$$
5\times(5+80)=425
$$

论文报告 COCO test-dev2015 上：

| 指标 | YOLOv2 |
|---|---:|
| AP@[0.5:0.95] | 21.6 |
| AP@0.5 | 44.0 |
| AP@0.75 | 19.2 |
| AP Small | 5.0 |
| AP Medium | 22.4 |
| AP Large | 35.5 |

注意：这里的 COCO AP 与 VOC mAP@0.5 不是同一个评价口径，不能直接横向比较。

---

## 16. 性能评估与实验结果

### 16.1 VOC 2007

论文 Table 3：

| 模型 / 输入尺寸 | mAP | FPS |
|---|---:|---:|
| YOLOv2 288×288 | 69.0 | 91 |
| YOLOv2 352×352 | 73.7 | 81 |
| YOLOv2 416×416 | 76.8 | 67 |
| YOLOv2 480×480 | 77.8 | 59 |
| YOLOv2 544×544 | 78.6 | 40 |

论文中这些结果使用同一组训练权重，只是推理时改变输入尺寸。测试硬件为原版 GeForce GTX Titan X。

### 16.2 YOLOv2 与 YOLOv1 对比

| 模型 | VOC 2007 mAP | FPS |
|---|---:|---:|
| YOLOv1 | 63.4 | 45 |
| YOLOv2 416×416 | 76.8 | 67 |
| YOLOv2 544×544 | 78.6 | 40 |

在论文对应的实验条件下，YOLOv2 相比 YOLOv1 提高了精度，同时在 416×416 输入时保持更高的速度。

但不要将这些历史 FPS 直接与现代 GPU 上的 YOLO11 或 YOLO26 比较。硬件、推理框架、输入分辨率和测试流程都不同。

### 16.3 Darknet-19 的效率

论文报告 Darknet-19 有 19 个卷积层、5 个 MaxPooling 层、5.58 Billion Operations，ImageNet Top-1 为 72.9%，Top-5 为 91.2%。

它的目标是用更高效的网络结构替代 YOLOv1 的特征提取网络，减少计算量并提升分类特征质量。

---

## 17. Ablation Study：YOLOv2 的改进到底贡献了什么？

论文 Table 2 展示了从 YOLOv1 逐步改进到 YOLOv2 的实验过程。

| 逐步加入的设计 | VOC 2007 mAP |
|---|---:|
| YOLO baseline | 63.4 |
| Batch Normalization | 65.8 |
| High Resolution Classifier | 69.5 |
| Fully Convolutional + Anchor Boxes | 69.2 |
| New Network | 69.6 |
| Dimension Priors | 74.4 |
| Direct Location Prediction | 75.4 |
| Passthrough | 76.8 |
| Multi-Scale Training | 78.6 |

### 17.1 Anchor Boxes 为什么让 mAP 略微下降？

论文在引入 Anchor Boxes 时，mAP 从 69.5 降到 69.2，而 Recall 从 81% 提升到 88%。

这说明 Anchor Boxes 增加了模型的候选框数量，让模型更容易找到潜在目标，但也引入了更多低质量预测。因此 Recall 上升并不一定意味着 mAP 同步上升。

### 17.2 Dimension Clusters 的提升

引入 K-means Anchor 之后，mAP 从 69.6 提升到 74.4，说明合适的 Anchor 尺寸先验可以显著降低边界框回归难度。

### 17.3 Direct Location Prediction 的提升

加入中心点 Sigmoid 约束后，mAP 从 74.4 提升到 75.4，说明限制中心点偏移能够提高 Anchor-based 回归的稳定性。

### 17.4 Passthrough 的提升

加入 Passthrough 后，mAP 从 75.4 提升到 76.8，说明较浅层的高分辨率特征对定位有帮助。

### 17.5 Multi-Scale Training 的提升

加入多尺度训练后，mAP 从 76.8 提升到 78.6，说明同一个模型能够从不同输入分辨率中学习更好的尺度适应能力。

---

## 18. 模型不足、失败案例与局限性

### 18.1 仍然只有一个检测输出尺度

Passthrough 将浅层细节融合到深层特征，但最终只有一个检测输出 Grid。它并不像后续 FPN / PAN 结构那样在多个尺度上分别预测目标。

因此，对于极小目标和密集目标，仍然会受到最终特征图分辨率的限制。

### 18.2 Anchor 依赖数据集分布

Anchor 是从训练集 GT 的宽高分布中聚类得到的。如果训练集与实际部署场景中的目标尺寸差异较大，原有 Anchor 可能不再合适。

例如训练数据中的目标大多是中大型物体，而部署时目标普遍只有十几个像素，原始 Anchor 的尺寸分布可能无法很好覆盖新场景。

因此在自定义数据集上训练 YOLOv2 时，应该重新统计 GT 的宽高分布，并重新聚类 Anchor。

### 18.3 一个 GT 只分配一个 Anchor

每个 GT 根据中心点分配到一个 Grid Cell，再选择一个最佳 Anchor。当多个 GT 中心落在同一个 Grid Cell 时，可能出现多个目标争用有限 Anchor 的情况。这会成为密集目标场景中的限制。

### 18.4 Anchor 数量与类别数量会影响输出规模

输出通道数：

$$
A(5+C)
$$

随着 Anchor 数量和类别数量增加，检测头输出规模也会增加。YOLO9000 为了控制输出规模，将 Anchor 从 5 个减少到 3 个。

### 18.5 仍然需要 NMS

YOLOv2 仍然会产生多个重叠候选框，需要 NMS 去除重复检测。它不是 NMS-free 的端到端集合预测模型。

### 18.6 WordTree 需要额外的类别体系

WordTree 能够解决不同数据集标签层级不一致的问题，但也增加了标签映射复杂度、类别树构建工作、不同数据集的标注兼容性问题，以及层级概率计算与训练逻辑。

它更适合有大量分类数据、且希望扩展检测类别范围的场景，不是所有普通检测任务都需要使用。

---

# 阶段 D：复现与应用（Reproduce & Apply）

## 19. 官方源码分析

作者官方仓库：

[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

建议重点阅读以下文件：

| 论文概念 | 源码文件 | 重点内容 |
|---|---|---|
| Darknet-19 分类网络 | **cfg/darknet19.cfg** | 19 层卷积分类网络 |
| YOLOv2 COCO | **cfg/yolov2.cfg** | 80 类、5 Anchor、425 输出通道 |
| YOLOv2 VOC | **cfg/yolov2-voc.cfg** | 20 类、5 Anchor、125 输出通道 |
| YOLOv2 Tiny | **cfg/yolov2-tiny.cfg** | 轻量化网络结构 |
| YOLO9000 | **cfg/yolo9000.cfg** | 9418 类、3 Anchor、WordTree |
| Region Layer | **src/region_layer.c** | Box Decode、Anchor Assignment、Objectness、Classification |
| Reorg Layer | **src/reorg_layer.c** | Space-to-Depth 的维度重排 |
| 配置解析 | **src/parser.c** | parse_region、parse_reorg |
| IoU | **src/box.c** | Box Intersection、Union 和 IoU |

### 19.1 Region Layer 的 Box Decode

Darknet **src/region_layer.c** 中：

~~~c
b.x = (i + x[index + 0*stride]) / w;
b.y = (j + x[index + 1*stride]) / h;
b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
~~~

对应：

$$
b_x=\frac{i+\sigma(t_x)}{S_w}
$$

$$
b_y=\frac{j+\sigma(t_y)}{S_h}
$$

$$
b_w=\frac{p_we^{t_w}}{S_w}
$$

$$
b_h=\frac{p_he^{t_h}}{S_h}
$$

源码中的 x、y 在 Region Layer 前向过程中已经经过 Logistic 激活；宽高则通过指数函数和 Anchor 尺寸解码。

### 19.2 Region Layer 的 Anchor Assignment

**src/region_layer.c** 中的训练逻辑：

1. 遍历每个 Grid Cell。
2. 遍历每个 Anchor。
3. 计算当前预测框与 GT 的最大 IoU。
4. 对普通负样本施加 No-Object Loss。
5. 如果 IoU 超过 Ignore Threshold，则跳过 No-Object Loss。
6. 遍历 GT，根据中心点确定 Grid Cell。
7. 根据 Anchor Shape IoU 选择最佳 Anchor。
8. 计算 Box、Objectness 和 Class 的训练梯度。

特别关注：

- get_region_box()
- delta_region_box()
- delta_region_class()
- forward_region_layer()

### 19.3 Reorg Layer

**src/reorg_layer.c** 中：

~~~c
if(reverse){
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c/(stride*stride);
}else{
    l.out_w = w/stride;
    l.out_h = h/stride;
    l.out_c = c*(stride*stride);
}
~~~

当 reverse=0、stride=2 时：

$$
[ B,C,H,W ]
\rightarrow
[ B,4C,H/2,W/2 ]
$$

这就是 Passthrough 使用的 Space-to-Depth 变换。

### 19.4 配置解析

**src/parser.c** 中的：

- parse_region()：解析 Anchor、类别数、坐标数、Softmax、Rescore、Loss 权重和阈值。
- parse_reorg()：解析 Reorg 的 Stride 和数据重排参数。

不要仅凭配置文件中的 num=5 就认为模型一定有 5 个有效 Anchor，还需要确认最后输出卷积通道数与：

$$
num\times(classes+coords+1)
$$

一致。

---

## 20. 论文、官方配置和源码之间的差异

### 20.1 YOLOv2 COCO 配置

官方 **cfg/yolov2.cfg**：

| 参数 | 配置 |
|---|---:|
| Input | 608×608 |
| Classes | 80 |
| Anchor 数量 | 5 |
| Coords | 4 |
| Filters | 425 |
| rescore | 1 |
| softmax | 1 |
| bias_match | 1 |
| object_scale | 5 |
| noobject_scale | 1 |
| class_scale | 1 |
| coord_scale | 1 |
| thresh | 0.6 |
| random | 1 |

输出通道验证：

$$
5\times(80+4+1)=425
$$

### 20.2 YOLOv2 VOC 配置

官方 **cfg/yolov2-voc.cfg**：

| 参数 | 配置 |
|---|---:|
| Input | 416×416 |
| Classes | 20 |
| Anchor 数量 | 5 |
| Filters | 125 |
| rescore | 1 |
| softmax | 1 |
| random | 1 |

输出通道：

$$
5\times(20+4+1)=125
$$

VOC 和 COCO 使用不同的 Anchor 聚类结果。

### 20.3 YOLOv2 Tiny

**cfg/yolov2-tiny.cfg** 是独立的轻量化结构，不是完整 YOLOv2 的同义名称。

它具有更少的卷积层、更少的通道数、不同的网络结构、更低的计算量，以及不同的精度与速度权衡。

当前仓库的 Tiny 配置还使用 rescore=0，与标准 YOLOv2 配置的 rescore=1 不同。因此不能把 Tiny 的训练行为直接当作标准 YOLOv2 的行为。

### 20.4 YOLO9000 配置

官方 **cfg/yolo9000.cfg**：

| 参数 | 配置 |
|---|---:|
| Input | 544×544 |
| Classes | 9418 |
| Anchor 数量 | 3 |
| Filters | 28269 |
| WordTree | data/9k.tree |
| COCO 映射 | data/coco9k.map |

输出通道：

$$
3\times(9418+5)=28269
$$

### 20.5 论文与源码配置不能混用

| 内容 | 论文 | 官方配置 |
|---|---|---|
| YOLOv2 Anchor 数量 | 5 | 标准配置 5 |
| VOC 输出通道 | 125 | 125 |
| COCO 输出通道 | 425 | 425 |
| 论文介绍的基础输入 | 416 | COCO cfg 默认 608，VOC cfg 416 |
| 多尺度训练 | 每 10 Batch 改变尺寸 | random=1 开启相应实现 |
| YOLO9000 Anchor 数量 | 3 | 3 |
| YOLO9000 类别数 | 9418 | 9418 |

**cfg/yolov2.cfg** 中的 width=608 是配置文件默认值，不代表论文所有实验都使用 608。论文 Table 3 的 288、352、416、480、544 是不同推理尺寸下的性能测试。

---

## 21. 复现 YOLOv2

### 21.1 获取代码和预训练权重

官方仓库：

~~~bash
git clone https://github.com/pjreddie/darknet
cd darknet
~~~

Darknet-19 预训练卷积权重：

~~~bash
wget https://pjreddie.com/media/files/darknet19_448.conv.23
~~~

该权重用于初始化 YOLOv2 Backbone。

### 21.2 VOC 训练示例

~~~bash
./darknet detector train \
    cfg/voc.data \
    cfg/yolov2-voc.cfg \
    darknet19_448.conv.23
~~~

其中：

- cfg/voc.data：VOC 数据集路径和类别配置。
- cfg/yolov2-voc.cfg：YOLOv2 VOC 网络配置。
- darknet19_448.conv.23：ImageNet 预训练卷积权重。

### 21.3 自定义数据集需要修改什么？

如果自定义数据集有 $C$ 个类别：

1. 修改数据集配置中的 classes=C。
2. 修改 Region Layer 中的 classes=C。
3. 修改最后一个检测卷积层的 filters。

标准 YOLOv2 使用 5 个 Anchor：

$$
\boxed{filters=5\times(C+5)}
$$

例如 6 类：

$$
filters=5\times(6+5)=55
$$

例如 20 类：

$$
filters=5\times(20+5)=125
$$

还需要根据自定义训练集重新计算 Anchor，而不是默认沿用 COCO 的 Anchor。

### 21.4 训练配置检查

- [ ] classes 是否与数据集类别数一致？
- [ ] 最后一个卷积层 filters 是否等于 $5(C+5)$？
- [ ] num 是否等于 Anchor 数量？
- [ ] anchors 是否与数据集匹配？
- [ ] coords 是否为 4？
- [ ] softmax 是否符合当前分类方式？
- [ ] rescore 是否与期望的 Objectness 监督一致？
- [ ] random=1 是否开启多尺度训练？
- [ ] 输入尺寸是否为 32 的倍数？
- [ ] 预训练权重是否与 Backbone 结构兼容？

---

## 22. 总结：YOLOv2 最值得掌握的知识

YOLOv2 并不是简单地在 YOLOv1 上增加几个卷积层，而是对目标检测的输出表达、特征提取和训练策略进行了一次系统升级。

### 22.1 Anchor Boxes 改变了 Bounding Box 预测方式

YOLOv1 直接预测框的位置和尺寸。

YOLOv2 以 Anchor 为先验，预测中心偏移和宽高尺度变化：

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y
$$

$$
b_w=p_we^{t_w}
$$

$$
b_h=p_he^{t_h}
$$

### 22.2 K-means 让 Anchor 更符合数据分布

通过：

$$
d=1-IoU
$$

对 GT 宽高进行聚类，得到更合适的尺寸先验。

### 22.3 Darknet-19 提高特征提取效率

19 个卷积层、5 个 MaxPooling 层、BN、Leaky ReLU 和 1×1 通道压缩共同构成高效 Backbone。

### 22.4 Passthrough 补充细粒度特征

通过 Reorg：

$$
26\times26\times64
\rightarrow
13\times13\times256
$$

再与：

$$
13\times13\times1024
$$

进行 Concat，得到 13×13×1280 的融合特征。

### 22.5 Multi-Scale Training 让同一模型适应不同输入尺寸

模型能够在不同输入分辨率下运行，提供速度与精度之间的可调节空间。

### 22.6 YOLO9000 将分类知识迁移到检测

WordTree 解决不同数据集标签层级不一致的问题，再通过检测与分类联合训练，让模型能够预测超过 9000 个类别。

---

## 参考资料

### 原始论文

1. [Joseph Redmon, Ali Farhadi — YOLO9000: Better, Faster, Stronger, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)
2. [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
3. [论文 PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

### 作者官方代码仓库

4. [pjreddie/darknet](https://github.com/pjreddie/darknet)
5. [Darknet-19 分类配置](https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg)
6. [YOLOv2 COCO 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg)
7. [YOLOv2 VOC 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg)
8. [YOLOv2 Tiny 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg)
9. [YOLO9000 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolo9000.cfg)
10. [Region Layer 实现](https://github.com/pjreddie/darknet/blob/master/src/region_layer.c)
11. [Reorg Layer 实现](https://github.com/pjreddie/darknet/blob/master/src/reorg_layer.c)
12. [配置解析实现](https://github.com/pjreddie/darknet/blob/master/src/parser.c)
13. [Bounding Box 与 IoU 实现](https://github.com/pjreddie/darknet/blob/master/src/box.c)

### YOLO9000 数据与类别树

14. [WordTree：9k.tree](https://github.com/pjreddie/darknet/blob/master/data/9k.tree)
15. [COCO 到 WordTree 的类别映射](https://github.com/pjreddie/darknet/blob/master/data/coco9k.map)
16. [ImageNet 到 WordTree 的类别映射](https://github.com/pjreddie/darknet/blob/master/data/inet9k.map)
17. [9k.names](https://github.com/pjreddie/darknet/blob/master/data/9k.names)

### 相关知识库文章

18. [YOLOv1：网络结构、Confidence、Loss 与完整训练推理解析](/articles/yolov1/)
