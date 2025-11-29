---
layout: page
permalink: /blogs/attention5/attention5/index.html
title: attention5
---

# 【attention3】deformable attention

## 前言

当我们直接查询deformable attention时，网络上会出现两种解读，往往会对初学者（比如我）产生很大的困惑。这些解读来自两篇论文：

“[Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159 "Deformable Transformers for End-to-End Object Detection")”

“[Deformable Attention Transformer](https://arxiv.org/abs/2201.00520 "Deformable Attention Transformer")”

前者是2020年的文章，简称deformable DETR；

后者是2022年的文章，简称DAT。

接下来，我会结合公式推导和这篇博客，

[   https://medium.com/@hanbrianlee/diving-into-the-difference-between-dat-deformable-attention-transformer-papers-approach-and-44bdf4ba5fa2](https://medium.com/@hanbrianlee/diving-into-the-difference-between-dat-deformable-attention-transformer-papers-approach-and-44bdf4ba5fa2 "   https://medium.com/@hanbrianlee/diving-into-the-difference-between-dat-deformable-attention-transformer-papers-approach-and-44bdf4ba5fa2")

尝试系统性地解释他们的应用与差异。

## deformable DETR

听这名字，我们就知道这是针对DETR的改进。开创性地用可变形注意力deformable attention替换我们熟悉的多头注意力，用在DETR上。

### DETR中的多头注意力

![](https://charmin161.github.io/blogs/attention5/image/image_KxSOtVwWXR.png)

DETR中的encoder和decoder都使用的是多头注意力机制，作者认为decoder中的多头注意力机制中的查询Q和每个K、V都做注意力运算，阻碍了训练速度。所以改成使用deformable attention，每个Q，只和一定范围内的KV，做运算。听着像不像swin transformer，不过deformable attention更进一步，这个和哪些KV做运算，是学习出来的。

#### MHA

先来一个我们熟悉的注意力机制

$$
z=\operatorname{Attention}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q \cdot K^{T}}{\sqrt{d_{k}}}\right) \cdot V
$$

![](https://charmin161.github.io/blogs/attention5/image/image_SuJuXZLoRc.png)

然后多几个头：

![](https://charmin161.github.io/blogs/attention5/image/image_mgdnFIxdTl.png)

然后还要接输出Wo,那计算方式就如下：

$$
z=Concat(z_1,z_2...z_M)\times{W_O}
$$

这个Wo的形状是h\*h的。

论文中的公式是这样的：

![](https://charmin161.github.io/blogs/attention5/image/image_D5o3YxxVQq.png)

怎么和我们之前写的不一样？拼接呢？怎么输入有两个？且听我一一解释

一是命名方式不同，比如这里的 $W_m$对应之前的 $W_O$，不过形状是 $h\times d$的，相当与将之前的 $W_O$拆分成了m个。

二是输入不同，这里的注意力机制，是cross attention， $z_q$对应Q中的一维， $x$对应KV。 $A_{mqk}$表示KQ运算出的注意力。

三是因为这里 $z_q$的大小为 $1\times d$，相当于原公式Q中的1行。而且所有的运算做了一个转置，即 $z_q$的大小为 $d\times 1$， $x_k$的大小为$d\times 1$，计算结果大小为$d\times 1$。

而concat的运算如下：

![](https://charmin161.github.io/blogs/attention5/image/微信图片_20240411215731_udDTG24ylb.jpg)

在上面那种形式中，就不是用concat，而是公式中的相加。

#### Deformable Attention

![](https://charmin161.github.io/blogs/attention5/image/image_VueMq4VrWU.png)

$$
\operatorname{DeformAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{p}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{k=1}^{K} A_{m q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}\left(\boldsymbol{p}_{q}+\Delta \boldsymbol{p}_{m q k}\right)\right]
$$

与多头注意力对比，有三点区别：

1、输入多了个 $p_q$，这是与查询相关的位置，表示在输入特征 $x$上的一个点。

2、每个点 $p_q$都可以找到多个相关的位移偏执 $\Delta p_{mqk}$，这些位移偏执也是用过线性层学习得到的。

3、这里的注意力矩阵A不再是通过QK做内积得到，而是直接对输入特征进行linear transformation。

针对图像处理中多尺度的特性，deformable attention也有相应的多尺度计算公式：

$$
\operatorname{MSDeformAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{p}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{l=1}^{L}\sum_{k=1}^{K} A_{m q k l} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}_{l}\left(\phi_l(\boldsymbol{p}_{q})+\Delta \boldsymbol{p}_{m q k l}\right)\right]
$$

虽然尺度不同，但query是一样的, $z_q$在每个尺度上都能找到对应的 $p_q$和相应的offset，所以输出的形状可以直接相加。同时，MSDeformaAttention在多尺度之间进行相加，有了类似FPN的效果。

### DAT

**要解决什么问题？**

![](https://charmin161.github.io/blogs/attention5/image/image_cZThaRqQ11.png)

**改进attention结构**，ViT算的太多；Swin Transformer 手工设计的注意力区域，会漏掉重要信息；DCN针对周围九个位置学习偏差，之后采样矫正过的特征位置，但分别训练，训练难度较大。

[DAT](https://zhida.zhihu.com/search?content_id=189045424\&content_type=Article\&match_order=1\&q=DAT\&zhida_source=entity "DAT")所有的 Q 会共享相同的感受野，但这些感受野会有学出来的位置偏差；

**使用什么方法？**

![](https://charmin161.github.io/blogs/attention5/image/image_2YqyreCRlc.png)

下采样得到参考点（reference points）→ 计算偏移点 → 通过双线性插值得到 $\widetilde x $→ 计算注意力机制

1、参考点是通过下采样得到的， $HW$大小的 $x$特征输入，得到的参考点大小为 $H/r\times H/r \times 2$，其中 $r$为缩放系数。

2、偏移点网络结构在右边的图b中显示，用了depth wise 卷积减少计算量

3、双线性插值，就是基于 $\Delta p$把四个角的特征加权求和

4、这里计算注意力机制时还加了个特殊的相对位置编码，根据图像的特性，也使用了双线性插值进行计算。

**与deformable detr的区别**

在笔者看来，这两就名字容易混淆，这计算方法不都不一样么？注意力计算方式不一样，偏执的计算网络不一样，位置编码也不一样。。。

Deformale detr作用在目标检测网络后期，特征提取之后，相当于网络的neck，不同的Q对应不同的V，可以关注不同的尺度和目标；

而DAT对标的是ViT，其实是可以作为detr的前面的主干网络的，所有的Q共享带偏执的KV，这点也很好理解，就算不同的角度分析图像，但图像中真正有意义的可能就那几个点。
