---
name: vision-model-study
description: 用于阅读、研究、解释和复现视觉领域的论文及模型。适用于目标检测、图像分类、语义分割、实例分割、姿态估计、自监督视觉表征、Vision Transformer、视觉基础模型、视觉语言模型等。统一按照理解模型、理解训练与推理、评价模型、复现与应用四个阶段展开，重点解释网络结构、模块计算、Tensor 数据流、Loss、实验和源码实现。
---

# 视觉模型论文阅读与模型技术解剖 Skill

## 一、目标与基本原则

本 Skill 用于系统性地研究视觉领域的论文和模型。

目标不是简单总结论文内容，而是帮助读者真正理解模型的设计思想、数学原理、网络结构、训练机制和实际实现，最终达到能够独立阅读源码、复现模型、分析模型不足并提出改进方案的程度。

适用领域包括但不限于：

- 目标检测：YOLO、DETR、RT-DETR、RF-DETR、DINO、DEIM 等。
- 图像分类：CNN、ResNet、EfficientNet、ViT、ConvNeXt 等。
- 图像分割：Semantic Segmentation、Instance Segmentation、SAM 等。
- 姿态估计：Human Pose Estimation、Keypoint Detection 等。
- 自监督视觉表征：DINO、DINOv2、DINOv3、MAE 等。
- 视觉基础模型：Vision Foundation Models。
- 视觉语言模型：CLIP、SigLIP、VLM 等。
- 新型视觉网络：Mamba、SSM 及 CNN-Transformer Hybrid 等。

### 1.1 核心原则

1. 以理解原理为主，不以罗列模型名称和模块名称为目的。
2. 优先阅读原始论文，并结合官方代码验证实际实现。
3. 重要模块必须解释具体计算过程，不能仅描述功能。
4. 重要网络必须追踪 Tensor 的维度变化和完整数据流。
5. 公式必须解释每个变量、计算步骤和数学含义。
6. 区分论文描述、源码实现和个人分析，不将推测当成论文结论。
7. 重点研究模型相较已有方法真正改变了什么，以及为什么有效。
8. 对实验结果、模型性能和创新性保持客观，不夸大论文贡献。
9. 论文没有公开的技术细节应明确说明，不允许自行补全。
10. 根据模型类型调整技术分析重点，不机械套用不相关的内容。

### 1.2 四阶段研究框架

所有完整的视觉模型论文阅读和模型技术分析，统一遵循以下四个阶段。

| 阶段 | 名称 | 核心问题 |
|---|---|---|
| A | 理解模型 | 模型是什么？为什么这样设计？ |
| B | 理解训练与推理 | 模型如何学习？如何运行？ |
| C | 评价模型 | 模型效果如何？改进是否有效？ |
| D | 复现与应用 | 如何自己实现、训练和部署？ |

具体研究内容分为以下 12 个维度。

---

# 二、阶段 A：理解模型（Understand the Model）

本阶段目标是理解模型的研究背景、任务定义、网络结构和完整前向传播过程。

## 01. 研究背景与问题定义（Research Motivation）

首先理解作者为什么提出这个模型。

需要回答：

1. 模型属于什么研究方向？
2. 主要解决什么问题？
3. 之前主流方法是什么？
4. 之前的方法存在什么不足？
5. 作者发现了什么技术瓶颈？
6. 作者提出了什么解决思路？
7. 模型最重要的创新是什么？
8. 哪些部分是作者原创，哪些部分来自已有研究？

需要将研究动机与具体技术设计对应起来。

建议使用以下表格：

| 原有问题 | 传统解决方案 | 存在的不足 | 当前模型的改进 |
|---|---|---|---|
| 问题 A | 方法 A | 不足 A | 改进 A |
| 问题 B | 方法 B | 不足 B | 改进 B |

不能只复述论文 Abstract，需要阅读 Introduction、Related Work 和 Method，理解技术改进的来龙去脉。

最终应形成研究背景、核心问题、技术瓶颈，以及创新点与解决方案的对应关系。

## 02. 任务定义与输入输出表示（Task Formulation）

明确模型处理的任务以及输入、输出的数学表示。

### 输入

- 输入数据类型。
- 图片尺寸及通道数。
- Tensor 排列方式：NCHW / NHWC。
- 输入归一化。
- 是否包含文本、深度、红外等其他模态。
- 是否使用多个尺度或多个视图。

### 输出

- 模型最终输出什么？
- 输出 Tensor 的维度是什么？
- 输出代表什么含义？
- 是否需要解码？
- 是否需要额外后处理？

### 标签与监督形式

- Bounding Box。
- Class Label。
- Mask。
- Keypoints。
- Image-level Label。
- Contrastive Pair。
- Teacher / Student Target。

根据具体任务解释标签如何表示。

例如目标检测：

$$
Y=\{(x_i,y_i,w_i,h_i,c_i)\}_{i=1}^{N}
$$

需要解释 N、坐标、宽高和类别分别代表什么。

注意：本节主要介绍任务定义、输入输出和标签语义。具体使用了哪些数据集、各数据集的图片数量、训练集划分等实验信息放在阶段 C。

## 03. 完整网络架构（Architecture Deep Dive）

这是整个模型研究中最重要的部分之一。要求达到能够结合论文和官方源码复现网络的程度。

### 3.1 整体架构

首先介绍网络整体组成。

例如目标检测模型：

- Backbone。
- Neck。
- Detection Head。
- Prediction Decoder。
- Post-processing。

例如 Transformer 检测器：

- Backbone。
- Feature Projection。
- Encoder。
- Decoder。
- Object Query。
- Prediction Head。

要求提供完整的网络结构图，标注模块名称、模块连接、输入输出、多尺度特征和关键 Tensor Shape。

### 3.2 模块内部结构

不能只解释模块名称，必须深入到具体运算。

对于每个重要模块，至少解释：

1. 模块的设计目的。
2. 输入 Tensor 的维度。
3. 输出 Tensor 的维度。
4. 内部由哪些运算组成。
5. 各运算的执行顺序。
6. 使用什么卷积、激活函数和归一化。
7. 是否包含残差连接、分支或特征融合。
8. 参数量和计算量如何计算。
9. 为什么使用这种设计。
10. 相较于已有模块有什么变化。

例如分析 C2f、C3k2、SPPF、AIFI、CCFM、MSAM 等模块时，必须展示其内部结构，而不是只介绍模块功能。

涉及 Attention 时，需要进一步解释：

- Q、K、V 的来源。
- 矩阵维度。
- Attention Score。
- Softmax。
- 特征加权。
- 多头 Attention。
- 计算复杂度。

涉及特殊算子时，需要解释其数学原理和具体计算过程。

### 3.3 作者为什么这样设计

对于每个重要改进，必须回答：

- 原有结构存在什么问题？
- 新模块针对什么问题？
- 为什么使用这种运算？
- 为什么放在网络的这个位置？
- 为什么不采用其他方法？
- 增加了多少参数和计算量？
- 是否增加了推理延迟？
- 作者通过什么实验验证其有效性？

不能仅凭模块名称推测设计目的，应结合论文和源码。

### 3.4 模块连接关系

重点分析：

- Concat。
- Add。
- Residual Connection。
- Self-Attention。
- Cross-Attention。
- Upsampling。
- Downsampling。
- Feature Fusion。
- Projection。

对于复杂网络，应区分训练时的结构、推理时的结构、辅助分支，以及可以在推理时移除的模块。

### 3.5 网络复杂度

尽可能分析：

- Parameters。
- FLOPs / MACs。
- Feature Map Size。
- Receptive Field。
- Activation Memory。
- Attention Complexity。

给出关键参数的计算过程，而不是只引用模型总参数量。

## 04. 单张图片的完整前向传播（Forward Pass）

这是理解模型数据流最重要的环节之一。必须尽可能使用一个具体输入，逐层追踪数据经过网络的变化。

### 4.1 输入图片

假设输入：

$$
X\in\mathbb{R}^{B\times H\times W\times 3}
$$

例如：

$$
B=1,\quad H=W=640
$$

需要说明图片进入网络后是否转换为：

$$
X\in\mathbb{R}^{1\times3\times640\times640}
$$

解释 NHWC 和 NCHW 的区别。

### 4.2 逐层追踪 Tensor

对重要网络层和模块建立表格：

| 网络位置 | 运算 | 输入 Shape | 输出 Shape | 作用 |
|---|---|---|---|---|
| Input | Image | ... | ... | 输入 |
| Conv | Convolution | ... | ... | 特征提取 |
| Backbone | Feature Block | ... | ... | 特征编码 |
| Neck | Feature Fusion | ... | ... | 多尺度融合 |
| Head | Prediction | ... | ... | 输出预测 |

所有维度必须依据具体模型配置和源码确认。不能为了让表格完整而编造通道数、层数或尺寸。

### 4.3 CNN 数据流

对于 CNN 模型，重点追踪：

- 每次卷积后的尺寸变化。
- Stride 对空间分辨率的影响。
- 通道数变化。
- 特征图下采样。
- 多尺度特征提取。
- Backbone 与 Neck 的连接。
- 检测头的输出维度。

### 4.4 Transformer 数据流

对于 Transformer 模型，重点追踪：

- CNN 或 ViT Backbone 输出。
- Feature Projection。
- Flatten。
- Tokenization。
- Positional Encoding。
- QKV 计算。
- Encoder。
- Object Query。
- Decoder。
- Prediction Head。

例如：

$$
[B,C,H,W]\rightarrow[B,HW,C]
$$

需要说明每一步的维度和计算意义。

### 4.5 特征语义变化

不仅解释 Tensor 的尺寸，还需要讨论不同阶段提取的特征。

例如浅层特征可能包含边缘、纹理和局部细节，中层特征可能包含局部结构和目标部件，深层特征可能包含高层语义信息。这些属于一般性解释，具体模型是否具有相应表现，需要结合特征可视化和实验分析。

### 4.6 最终输出

必须解释：

- 网络输出的 Tensor。
- 输出维度。
- 每个维度代表什么。
- 如何转换成最终预测结果。

对于目标检测，说明分类分数、Bounding Box、Objectness、Query 等信息的来源。

最终应提供完整的数据流图和 Tensor Shape 表格。

---

# 三、阶段 B：理解训练与推理（Understand Training & Inference）

本阶段目标是理解模型如何学习，以及训练和推理过程有什么区别。

## 05. 训练机制与优化过程（Training Pipeline）

### 5.1 参数初始化

需要研究：

- 是否使用预训练权重？
- Backbone 是否使用 ImageNet 或其他数据集预训练？
- Head 是否随机初始化？
- 使用 Xavier、Kaiming 等初始化方法吗？
- 是否冻结部分参数？
- 是否存在特殊的 Bias 初始化？
- 是否使用 Teacher-Student？
- Teacher 和 Student 如何初始化？
- Teacher 参数如何更新？

对于自监督学习、蒸馏和基础模型，必须重点研究权重初始化和参数更新机制。

### 5.2 Loss Function

必须解释 Loss 的具体计算过程，不能只写 Loss 名称。

对于每一个 Loss，至少说明：

1. 数学公式。
2. 每个变量的含义。
3. 输入是什么。
4. 输出是什么。
5. 如何计算。
6. 为什么使用该 Loss。
7. 相较于传统 Loss 有什么变化。
8. 在整体训练中占多大权重。

例如目标检测：

$$
L=L_{box}+\lambda_1L_{cls}+\lambda_2L_{dfl}
$$

需要分别解释 Box Loss、Classification Loss、Distribution Focal Loss、Loss 权重和多尺度 Loss 聚合。

如果是 DETR，还需要解释 Hungarian Matching 如何决定参与 Loss 计算的预测与 GT。

如果存在辅助 Loss、Denoising Loss、Distillation Loss，需要单独说明。

### 5.3 正负样本分配（Label Assignment）

目标检测模型必须重点分析。

需要回答：

- Anchor 如何生成？
- GT 如何与预测匹配？
- 如何确定 Positive Sample？
- 如何确定 Negative Sample？
- 是否存在 Ignore Sample？
- 一个 GT 可以匹配多少个预测？
- 是否使用 One-to-one 或 One-to-many？
- 匹配代价如何计算？

重点分析相关方法：

- IoU-based Assignment。
- ATSS。
- PAA。
- OTA。
- SimOTA。
- TAL。
- Hungarian Matching。
- One-to-one Matching。

必须解释具体匹配过程，必要时通过一个小规模数值例子展示。

### 5.4 数据增强

需要研究：

- Mosaic。
- MixUp。
- HSV。
- Random Resize。
- Random Crop。
- Horizontal Flip。
- Copy-Paste。
- Multi-scale Training。

说明每种增强如何改变输入图片以及对应标签。同时关注增强策略是否存在关闭时机、不同训练阶段是否采用不同增强方式。

### 5.5 优化器和训练参数

至少记录：

| 项目 | 需要研究的内容 |
|---|---|
| Optimizer | SGD、Adam、AdamW 等 |
| Learning Rate | 初始学习率 |
| Scheduler | Cosine、Step 等 |
| Batch Size | 全局 Batch Size |
| Epoch | 总训练轮数 |
| Weight Decay | 权重衰减 |
| Momentum | 动量 |
| EMA | 是否使用 |
| Mixed Precision | FP16、BF16 等 |
| Gradient Clipping | 是否使用 |
| Warmup | 是否存在预热阶段 |

对于特殊优化器，需要进一步解释内部更新机制。

### 5.6 完整训练流程

绘制训练流程图：

Dataset
→ Data Augmentation
→ Forward
→ Label Assignment / Matching
→ Loss
→ Backpropagation
→ Optimizer Update

需要区分前向传播、Loss 计算、梯度反向传播、参数更新、EMA 更新和 Teacher 更新。

如果存在多阶段训练、冻结参数、蒸馏等机制，需要单独绘制流程。

## 06. 推理过程与后处理（Inference Pipeline）

训练过程和推理过程可能不同，必须分别解释。

### 6.1 输入预处理

- Resize。
- Letterbox。
- Normalize。
- Padding。
- RGB / BGR。
- 动态输入尺寸。

### 6.2 前向推理

解释推理时实际执行哪些网络模块。

重点关注训练专用模块是否移除、辅助分支是否移除、One-to-one Head 是否单独使用、Query 数量是否发生变化，以及是否存在多尺度推理。

### 6.3 输出解码

根据模型类型解释输出如何转换为最终结果。

例如：

Anchor-based：

$$
Box=f(Anchor,Prediction)
$$

Anchor-free：

$$
Box=f(FeaturePoint,Prediction)
$$

DETR：

$$
Box=Sigmoid(MLP(Query))
$$

必须解释坐标如何还原到原始图像。

### 6.4 后处理

根据模型实际情况分析：

- NMS。
- Soft-NMS。
- DIoU-NMS。
- Score Threshold。
- Top-K。
- Clipping。
- Query Selection。

对于 NMS，必须解释：

1. 为什么需要 NMS？
2. IoU 如何计算？
3. 置信度如何排序？
4. 哪些框被保留？
5. NMS Threshold 如何影响结果？
6. NMS 的时间复杂度和实际开销。

对于 NMS-free 模型，必须说明其如何避免冗余预测，以及是否仍存在 Top-K 或其他候选筛选。

### 6.5 训练和推理差异

建议使用对比表：

| 项目 | Training | Inference |
|---|---|---|
| 数据输入 | ... | ... |
| 网络分支 | ... | ... |
| Loss | ... | 不计算 |
| 辅助监督 | ... | ... |
| Query | ... | ... |
| 后处理 | ... | ... |

### 6.6 导出与部署差异

如果模型涉及工程部署，分析：

- PyTorch。
- ONNX。
- TensorRT。
- FP32 / FP16 / INT8。
- 动态输入。
- 算子支持。
- 输出 Tensor 差异。
- NMS 是否被导出到计算图。

## 07. 可解释性与特征可视化（Interpretability）

首先检查原论文是否提供可视化实验。

重点关注：

### 7.1 Feature Map

观察 Backbone、Neck 等不同层的特征图，分析网络提取的信息。

### 7.2 Attention Map

观察 Attention 模块重点关注的空间区域和特征关系。

### 7.3 Grad-CAM

分析模型某个预测结果主要依赖哪些图像区域。

### 7.4 Feature Embedding

通过 t-SNE、UMAP 等方法观察不同类别的特征分布。

### 7.5 Multi-scale Feature Visualization

对比 P3、P4、P5 等不同尺度的特征表达。

### 7.6 可解释性实验要求

如果论文提供了相关图片：

- 解释图片表达的内容。
- 解释可视化方法。
- 说明作者希望证明什么。
- 分析这些可视化是否足以支持作者的结论。

如果论文没有提供：

- 不要虚构论文中的可视化结果。
- 可以提出自行复现的可视化方案。
- 区分论文原始实验和额外分析建议。

注意：Attention Map、Grad-CAM 等可视化不能单独作为模型因果解释的充分证据。

---

# 四、阶段 C：评价模型（Evaluate the Model）

本阶段目标是评价模型性能、分析创新有效性，并判断模型的适用范围。

## 08. 数据集与实验设置（Dataset & Experimental Protocol）

这里重点研究论文实际使用的数据集和实验条件。

### 8.1 数据集

记录：

- 数据集名称。
- 图片数量。
- 类别数量。
- 标注数量。
- 图片分辨率。
- 目标尺寸分布。
- 目标密度。
- 训练集、验证集和测试集划分。

### 8.2 预训练数据

需要单独记录：

- Backbone 预训练数据。
- 自监督预训练数据。
- 蒸馏数据。
- 额外训练数据。
- 无标注数据。

区分检测训练数据与预训练数据。

### 8.3 实验配置

记录：

- 输入尺寸。
- Batch Size。
- Epoch。
- GPU 型号。
- GPU 数量。
- 优化器。
- 学习率。
- 预训练权重。
- 数据增强。
- 推理框架。

### 8.4 数据集适用性

分析：

- 数据集能否代表目标应用场景？
- 是否存在类别不平衡？
- 是否存在长尾分布？
- 是否存在严重遮挡？
- 小目标占比是多少？
- 目标密度是否足够高？

不能只罗列数据集名称，还要说明数据集特点对实验结果的影响。

## 09. 性能评估与实验结果（Performance Evaluation）

### 9.1 精度指标

根据任务选择：

| 任务 | 主要指标 |
|---|---|
| 目标检测 | AP、AP50、AP75、APS、APM、APL、Precision、Recall |
| 图像分类 | Top-1、Top-5、Accuracy、F1 |
| 语义分割 | mIoU、Dice、Pixel Accuracy |
| 实例分割 | Box AP、Mask AP |
| 姿态估计 | AP、OKS、关键点误差 |

解释指标的定义、公式和适用条件。

### 9.2 模型复杂度

- Parameters。
- FLOPs / MACs。
- Model Size。
- Activation Memory。
- Peak Memory。

### 9.3 训练效率

- 单 Epoch 训练时间。
- 总训练时间。
- Images/sec。
- GPU 利用率。
- 显存占用。
- 不同 Batch Size 下的效率。

如果论文没有提供训练时间，不允许根据 Epoch 数量直接编造。

### 9.4 推理效率

- PyTorch FPS。
- ONNX FPS。
- TensorRT FPS。
- FP32 / FP16 / INT8。
- Batch Size=1 Latency。
- Throughput。
- 不同输入尺寸的推理时间。
- 端到端延迟。

### 9.5 公平比较

比较不同模型时，尽量统一数据集、输入分辨率、训练数据、预训练条件、训练预算、硬件、推理框架、精度模式和测量方式。

不能将不同测试条件下的 AP 和 FPS 直接作为严格的横向排名。

## 10. 与前代模型对比及创新分析（Novelty & Ablation）

### 10.1 前代模型对比

明确当前模型相较于之前的方法发生了什么变化。

建议使用：

| 比较维度 | Previous Model | Current Model | 改进目的 |
|---|---|---|---|
| Backbone | ... | ... | ... |
| Neck | ... | ... | ... |
| Head | ... | ... | ... |
| Feature Fusion | ... | ... | ... |
| Label Assignment | ... | ... | ... |
| Loss | ... | ... | ... |
| Inference | ... | ... | ... |

不能仅列出模块名称，必须解释技术原理。

### 10.2 创新点分类

区分网络架构创新、模块结构创新、特征提取创新、训练机制创新、Loss 创新、匹配机制创新、推理机制创新和工程部署创新。

### 10.3 Ablation Study

重点研究论文的消融实验。

需要分析：

1. 单独增加模块 A 的效果。
2. 单独增加模块 B 的效果。
3. 多个模块组合的效果。
4. 不同参数设置的影响。
5. 模块之间是否存在协同作用。
6. 精度提升是否伴随着计算成本增加。

不能仅根据最终模型的 AP 提升，就断言每个新增模块都有效。

### 10.4 证据区分

明确区分作者声称的创新、论文实验能够证明的创新、官方代码实际实现的创新，以及独立复现后验证的创新。

## 11. 模型不足、失败案例与泛化能力（Limitations & Robustness）

### 11.1 模型自身局限

分析小目标、密集目标、遮挡、复杂背景、类别不平衡、计算量、推理延迟、训练成本和部署限制。

### 11.2 失败案例

寻找论文中的失败案例，或设计额外测试。

重点分析漏检、误检、重复检测、定位偏差、类别混淆、目标遮挡和极端尺寸。

### 11.3 泛化能力

研究跨数据集泛化、跨场景泛化、跨设备泛化、不同图像分辨率、不同光照和不同天气。

### 11.4 鲁棒性

根据模型应用分析：

- Gaussian Noise。
- Blur。
- Color Shift。
- Occlusion。
- Domain Shift。
- Adversarial Perturbation。

不能将单一测试集上的高 AP 等同于模型具有良好的泛化能力。

### 11.5 综合评价

最终明确模型最擅长解决什么问题、不适合什么场景、相较于已有方法真正进步在哪里，以及还存在哪些研究机会。

---

# 五、阶段 D：复现与应用（Reproduce & Apply）

本阶段目标是将论文中的方法转化为可以运行、训练和部署的工程实现。

## 12. 官方代码分析、模型复现与部署（Code & Reproduction）

### 12.1 资料来源

优先检查：

1. 原始论文。
2. 官方 GitHub 仓库。
3. 官方配置文件。
4. 官方训练代码。
5. 官方模型权重。
6. Hugging Face 官方模型页面。
7. 官方部署文档。

区分官方实现与社区二次实现。

### 12.2 源码结构

重点定位 Model Definition、Backbone、Neck、Encoder、Decoder、Head、Matcher、Criterion / Loss、Dataset、Data Augmentation、Training Script、PostProcessor 和 Export Script。

### 12.3 源码与论文对应

将论文中的模块与源码中的实际实现对应起来。

例如：

| 论文中的概念 | 源码位置 | 实现内容 |
|---|---|---|
| Backbone | ... | ... |
| Feature Fusion | ... | ... |
| Query Selection | ... | ... |
| Matcher | ... | ... |
| Loss | ... | ... |
| PostProcessor | ... | ... |

必须沿着 forward 路径追踪数据，不能仅通过文件名判断模块作用。

### 12.4 版本差异

如果论文、GitHub 和模型权重配置不一致：

- 分别记录不同版本。
- 明确采用的配置。
- 说明差异。
- 不将不同版本的参数和结构拼在一起。
- 不将论文描述直接当作当前源码的实际行为。

### 12.5 模型复现

记录环境依赖、数据集准备、模型配置、预训练权重、训练命令、关键超参数、训练结果和验证结果。

将论文结果与实际复现结果进行对比，分析差异原因。

如果没有实际执行训练，必须明确标注为复现方案，而不是已经复现成功。

### 12.6 模型部署

根据应用需要研究 PyTorch、ONNX、TensorRT、C++、FP16、INT8、动态输入、推理算子支持和后处理实现。

分析模型在部署过程中的结构变化、精度变化和性能变化。

### 12.7 工程应用

结合具体应用场景分析：

- 是否适合当前硬件？
- 是否满足实时性要求？
- 是否需要量化？
- 是否需要蒸馏？
- 是否需要裁剪？
- 是否需要重新训练？
- 是否适合现有数据集？

最终形成可操作的模型使用和改进建议。

---

# 六、不同视觉模型的专项分析要求

四阶段框架适用于所有视觉模型，但不同任务需要增加专门的技术分析。不能机械地将目标检测模型的 Anchor、NMS 等内容应用到图像分类模型中。

## 6.1 目标检测模型

必须重点研究：

- Anchor-based / Anchor-free。
- Backbone / Neck / Head。
- FPN / PAN / BiFPN。
- Feature Pyramid。
- Label Assignment。
- Positive / Negative Sample。
- Box Regression。
- Classification Loss。
- IoU / GIoU / CIoU。
- DFL。
- NMS / Soft-NMS。
- One-to-one / One-to-many。
- Hungarian Matching。
- NMS-free。
- Prediction Decode。

## 6.2 图像分类模型

重点研究：

- Backbone。
- Global Average Pooling。
- Classification Head。
- Classifier。
- Softmax。
- Cross Entropy。
- Label Smoothing。
- Class Imbalance。
- Transfer Learning。
- Feature Embedding。

## 6.3 分割模型

重点研究：

- Semantic / Instance / Panoptic Segmentation。
- Pixel-level Prediction。
- Mask Head。
- Mask Decoder。
- Mask Loss。
- Upsampling。
- Feature Resolution。
- Mask Matching。
- Dice Loss。
- mIoU。

## 6.4 Pose 模型

重点研究：

- Keypoint Representation。
- Keypoint Encoding。
- Heatmap / Regression。
- Keypoint Visibility。
- Flip Augmentation。
- OKS。
- Keypoint Loss。
- 不同类别关键点定义。

## 6.5 自监督视觉模型与 Foundation Model

重点研究：

- Teacher / Student。
- EMA。
- Momentum Update。
- Multi-crop。
- Self-distillation。
- Contrastive Learning。
- Masked Image Modeling。
- Register Token。
- Feature Alignment。
- Representation Collapse。
- Pretraining Dataset。
- Pretraining Schedule。
- Downstream Transfer。

尤其需要解释预训练模型的知识如何迁移到检测、分类、分割等下游任务。

## 6.6 Vision Transformer

重点研究：

- Patch Embedding。
- Positional Encoding。
- QKV。
- Multi-head Attention。
- FFN / MLP。
- LayerNorm / RMSNorm。
- Residual Connection。
- Token 数量。
- Attention Complexity。
- Global / Local Attention。
- Window Attention。

## 6.7 Mamba / State Space Model

重点研究：

- State Space Model。
- 状态空间方程。
- A、B、C、D 参数。
- Discretization。
- Selective Scan。
- 状态更新。
- 序列建模。
- 计算复杂度。
- 与 CNN / Transformer 的差异。

## 6.8 Vision Language Model

重点研究：

- Image Encoder。
- Text Encoder。
- Tokenization。
- Projection Layer。
- Cross-modal Alignment。
- Contrastive Loss。
- Image-Text Pair。
- Visual Token。
- Multimodal Fusion。
- 训练数据与预训练策略。

---

# 七、资料检索与证据要求

研究论文和模型时，不能只依赖论文摘要或搜索结果。

按照以下优先级收集资料：

1. 原始论文及其补充材料。
2. 官方 GitHub 仓库。
3. 官方配置文件和源码。
4. 官方模型权重及模型卡。
5. 官方技术文档。
6. 独立复现论文和技术报告。
7. 论坛、博客和社区经验。

### 7.1 论文核对

重点检查 Abstract、Introduction、Related Work、Method、Experiments、Ablation Study、Limitations 和 Supplementary Material。

### 7.2 源码核对

重点追踪模型注册入口、配置文件、网络初始化、Forward、Loss、Matcher、PostProcessor、Training Script 和 Export Script。

### 7.3 事实与推断区分

技术文章中需要明确区分：

| 信息类型 | 说明 |
|---|---|
| 论文事实 | 作者在论文中明确说明 |
| 源码事实 | 官方代码中的实际实现 |
| 实验事实 | 论文或实际实验公布的数据 |
| 技术解释 | 根据计算原理进行的解释 |
| 分析推断 | 根据现有证据推导出的结论 |
| 未知信息 | 论文和源码没有公开的内容 |

不能将技术推断描述成作者明确提出的设计目的，不能将未经验证的社区观点当作实验事实。

---

# 八、回答与文章输出规范

## 8.1 普通模型解释与论文阅读

用户要求阅读论文、总结论文或解释模型时：

- 默认使用四阶段框架组织回答。
- 根据问题深度决定各部分展开程度。
- 完整论文解读应尽可能覆盖所有适用的 12 个维度。
- 用户只问某个模块时，直接深入对应模块，不必机械输出全部章节。
- 使用中文讲解，专业术语和模型模块名称保留英文。
- 先介绍整体原理，再讲内部计算。
- 公式、维度和具体数值示例应配合解释。
- 适当使用表格和 Mermaid 图。
- 对资料缺失、版本差异和未经验证的内容明确说明。

## 8.2 技术深度要求

对于用户要求详细讲解的模型，不能只回答模块是什么、模块有什么作用、模型有哪些创新、指标提升了多少。

必须进一步解释：

- 模块内部具体如何计算。
- 输入和输出是什么。
- 每一步 Tensor 维度如何变化。
- 参数如何更新。
- Loss 如何计算。
- 不同模块之间如何交互。
- 作者为什么这样设计。
- 有什么优缺点。
- 是否能够从源码复现。

## 8.3 文章编写

当用户要求将模型研究内容整理成知识库文章或 Markdown 文档时，必须同时遵循：

`skills/knowledge-article.md`

不得另行创造一套与知识库冲突的 Markdown 规范。

文章要求：

- 使用中文。
- 结构完整。
- 技术内容自洽。
- 保留关键公式、推导、表格和代码。
- 使用标准 Markdown。
- 重要网络结构和训练流程优先使用 Mermaid。
- 公式使用行内和独立 LaTeX 数学公式。
- 参考资料指向真实论文和官方源码。
- 不省略关键实现细节。

当用户要求输出可直接复制的完整 Markdown 原文时：

1. 全文放在一个独立 Markdown 代码框中。
2. 外层代码框使用四个反引号。
3. 内部代码块使用三个反引号。
4. 不拆分多个代码框。
5. 代码框之外不重复正文，只保留必要的简短说明。

如果直接向知识库仓库写入文章，文件中应保存正常的 Markdown 原文，不要将外层四反引号包装写入实际文件。

## 8.4 模型研究的标准图表

对于完整模型研究，优先提供以下三张核心图：

### 图一：完整网络架构图

包含输入、Backbone、Neck、Head、Encoder / Decoder、关键模块和输出。

### 图二：完整训练流程图

包含数据输入、数据增强、Forward、Matching / Assignment、Loss、Backpropagation 和 Optimizer Update。

### 图三：单张图片前向传播图

包含输入 Tensor、各网络阶段、特征图尺寸、通道变化、关键计算和最终预测。

复杂模型可以额外绘制模块内部结构图。

必须区分论文原图、源码结构图和自行绘制的解释性示意图。

---

# 九、最终研究质量检查清单

完成一篇完整的视觉模型研究后，检查以下内容：

## 阶段 A：理解模型

- [ ] 是否解释研究背景和技术瓶颈？
- [ ] 是否明确任务定义和输入输出？
- [ ] 是否绘制完整网络架构？
- [ ] 是否解释关键模块内部结构？
- [ ] 是否解释作者为什么这样设计？
- [ ] 是否分析模块之间的数据流？
- [ ] 是否追踪单张图片的完整前向传播？
- [ ] 是否给出关键 Tensor Shape？

## 阶段 B：理解训练与推理

- [ ] 是否解释权重初始化？
- [ ] 是否说明预训练权重来源？
- [ ] 是否解释 Loss 公式和计算过程？
- [ ] 是否解释正负样本分配？
- [ ] 是否分析数据增强？
- [ ] 是否记录优化器和学习率策略？
- [ ] 是否绘制完整训练流程？
- [ ] 是否解释推理阶段的实际网络结构？
- [ ] 是否解释解码和后处理？
- [ ] 是否说明训练与推理的差异？
- [ ] 是否分析可视化结果？

## 阶段 C：评价模型

- [ ] 是否说明数据集和实验设置？
- [ ] 是否区分训练数据和预训练数据？
- [ ] 是否解释评价指标？
- [ ] 是否分析精度、参数量和计算量？
- [ ] 是否分析训练和推理效率？
- [ ] 是否与前代模型进行对比？
- [ ] 是否分析消融实验？
- [ ] 是否区分作者声称和实验验证的创新？
- [ ] 是否分析模型不足和失败案例？
- [ ] 是否研究泛化能力和鲁棒性？

## 阶段 D：复现与应用

- [ ] 是否找到官方代码仓库？
- [ ] 是否定位关键源码文件？
- [ ] 是否将论文模块与源码对应？
- [ ] 是否确认配置和权重版本？
- [ ] 是否提供可执行的复现方案？
- [ ] 是否区分方案与实际复现结果？
- [ ] 是否分析 ONNX / TensorRT 等部署方式？
- [ ] 是否评估实际应用的适用性？

---

# 十、最终原则

本 Skill 的最终目标是建立一套可以迁移到不同视觉模型上的研究方法。

不要求每个模型都机械地包含所有内容，但完整模型研究必须遵循四阶段逻辑：

**阶段 A：理解模型**

知道模型解决什么问题，网络如何构建，数据如何流动。

**阶段 B：理解训练与推理**

知道模型如何学习、参数如何更新、Loss 如何计算，以及推理时如何产生结果。

**阶段 C：评价模型**

知道模型性能如何、创新是否有效、与已有方法相比有什么优势，以及还存在哪些不足。

**阶段 D：复现与应用**

能够阅读源码、复现主要实验，并将模型应用到实际工程中。

最终达到：不仅能够读懂论文，还能够独立解释网络结构、推导关键公式、追踪 Tensor、实现核心模块、复现训练过程，并基于实验发现模型的不足和改进方向。
