---
layout: page
permalink: /cn/index.html
title: 闵雨松的个人主页
---

# 中文简历

> **核心亮点**
：退伍军人 | 东南大学本硕（控制科学与工程） | 算法工程师 | 专注模型训练与底层算法优化

**坐标**：上海 | 邮箱：qiguai1619908@163.com

[知乎主页：YoursMin](https://www.zhihu.com/people/bu-si-zhu) | [github 主页：charmin161](https://github.com/charmin161)



***

## 核心技能

#### 编程语言与工具

Python/C++/CUDA | PyTorch/TensorRT | Git/Docker/OpenCV

#### 核心技术领域

Tensor Core 硬件优化 | 低精度模型量化（INT8/INT4/FP8/FP4） | 多目标跟踪（DeepSORT/ByteTrack）

#### 学术与工程能力

论文发表（CCF 会议 + SCI 期刊） | 模型训推全流程 | 视觉算法 | 数值计算优化

***

## 工作经历

#### 中兴通讯 - 算法工程师

2023.07-2025.12   &#x20;

1、负责tensor core 底层矩阵乘算法设计与算法支持

2、低精度模型训推方案探索与验证

3、参与小模型精调，为模型算子开发工作提供算法支持，解决模型部署中的性能瓶颈问题

4、负责车联网项目视频算法工作，实现复杂场景下多目标精准定位，满足实际业务落地需求

***

## 项目经历

#### 底层矩阵乘算法优化项目 

2024.10 - 2025.11

* **项目背景**：针对底层不同数据格式乘累加需求，设计高效矩阵运算方案

* **核心职责**：基于业界领先方案与自身硬件限制，设计矩阵乘累加具体方案；设计测试用例并编写python端实现代码，协助开发对齐。

* **项目成果**：交付算法核心方案，满足底层开发需求

#### 低精度模型训推方案研发

2024.8 - 2025.11

* **项目背景**：解决大模型训练推理时显存占用高、推理延迟大的痛点，探索超低比特量化技术落地可行性

* **核心职责**：调研 INT8/INT4/FP8/FP4 量化方案，设计量化训练策略；搭建测试基准，验证模型精度与性能平衡关系

* **项目成果**：充分验证低比特精度在训推方面的性能瓶颈与优化方案，为底层开发提供支持

#### 车联网路口多目标跟踪系统开发

2023.8 - 2024.10

* **项目背景**：面向智能交通场景，需实现路口车辆、行人等目标的实时定位与轨迹追踪

* **核心职责**：基于 YOLO+DeepSORT 算法构建检测 - 跟踪一体化框架，融合 ReID 技术解决目标遮挡问题；优化算法适配边缘设备部署

* **项目成果**：车辆与行人定位轨迹准确率达到99%以上，实现单一路口多台摄像头协同运算，满足车联网实时性要求



***
## 教育

### 东南大学

2020-2023  控制科学与工程 - 硕士
* 竞赛得奖： 2021.05      “华为杯”第十八届中国研究生数模竞赛二等奖（国奖）
* 硕士论文： 基于深度学习的高速公路车流量监测系统研究

### 东南大学

2014-2020 自动化 - 学士
* 服役经历：2017-2019 入伍参军

***

## 论文发表

##### [*A Vehicle Counting and Road Condition Analysis System Based on Multiple Object Tracking*](https://link.springer.com/chapter/10.1007/978-981-19-6203-5_42 "A Vehicle Counting and Road Condition Analysis System Based on Multiple Object Tracking")

第18届中国智能系统会议 (CISC2022)
**Yusong Min** and Junyong Zhai



##### [*A Vehicle Comparison and Re-identification System Based on Residual Network*](https://doi.org/10.3390/machines10090799 "A Vehicle Comparison and Re-identification System Based on Residual Network")[ ](https://doi.org/10.3390/machines10090799 " ")

Machines
Weifen Yin, **Yusong Min** and Junyong Zhai

##### [基于深度学习的高速公路车流量监测系统研究](https://kns.cnki.net/kcms2/article/abstract?v=-PyPURV5YK0PinKnHL2gbLtje8gQVJ6S7iTOYRZTFSGfB_7kcfEzJpqlpiWwp8e6lFh5H-I85ZM1ZlhwJwTRpR6pM9cdRP4xkgiur2G3STML1ch2xxm7HVkXRJ-9L115Q_rWoIe5xkxbixtlbVX6dqKzXJBJWeTdLdcQ-AFIOKHWsIkfmKYXi32T6sB5wfMh&uniplatform=NZKPT&language=CHS)
硕士论文
**闵雨松** (指导教师: 翟军勇)
