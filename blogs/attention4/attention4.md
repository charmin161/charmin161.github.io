---
layout: page
title: attention4
---

# 【attention4】可训练稀疏注意力MoBA & NSA & DSA 

之前的文中介绍了MLA的降秩操作，巧妙地利用attention结构中的冗余性，通过降秩的方案，减少KV cache，加快推理速度。

这次的两篇文章，kimi推出的MoBA（Mixture of Block Attention）和deepseek推出的NSA（Native Sparse Attention），都利用了full attention中的冗余性，使用稀疏的注意力方案，可以说是把MLP那边的思想，在attention里又用了一遍。虽然知道冗余，但要**稀疏哪里**呢？在计算 $QK^T$的过程中，如果不算，就不知道哪里稀疏，但算了就没必要再稀疏了（没有减少计算量），顶多可以减少后面V的计算量。所以如果可以在训练过程中，**从数据中学到注意力的稀疏性**，就很妙了。这两篇文章的核心卖点就是这个。

本文首先简单介绍这两篇内容的主要思想，再提炼两篇内容的共同观点，并比较一下不同。（DSA是deepseek后续2025.09在DeepseekV3.2中对NSA的升级，放在NSA一起讲）

### 1、MoBA

1）分块稀疏 将上下文划分为多个块，每个查询学习最相关的KV

考虑普通的注意力：

$$
Attn(q,K,V) = Softmax(qK^T)V
$$

设KV的长度为N，MoBA的query只查询其子集 $I\subset[N]： $

$$
MoBA(q,K,V) = SoftMax(qK[I]^T)V[I]
$$

这里的关键是I的选取。（当然这里I可等于N，这样就是全注意力了，调参党又喜（bushi）

对N使用**块划分**，划分为n个块，每个块长为B:

$$
N = [I_1,I_2,...I_i,...I_n]
\\ I_i = [(i-1)\times B +1, i \times B]
$$

2）无参数门控 使用topk 选择每个Q最相关的块

使用**门控**技术进行选取（直接用MoE的ToPK）:

$$
s_i = <q,mean\_pool(K[I_i])>\\
g_i=\left\{
\begin{aligned}
1 && s_i \in TopK(s_j | j \in [n]) \\
0 && otherwise 
\end{aligned}
\right. \\
I = {\cup}_{g_i > 0}I_i
$$

可以看到，这里减少计算量的核心，是对 $K[I_i]$进行mean\_pool来降维，通过这个指标来判断q和K的相关性，是MoE的思想。

PS：让人产生了一种给注意力计算注意力的感觉？

### 2、NSA

NSA的动机也是筛选出更少的KV，其名字，native sparse指的是预训练的时候就是稀疏的注意力结构，使用**KV序列**的稀疏化。
为什么可以使用KV序列呢，因为从attention的计算公式可以推导出，k和v的词元数量不影响输出o的维度，其实也相当于一个降维。

目前针对长上下文的方案有以下这些：
1、线性注意力，讲过去的状态压缩成隐状态
2、窗口注意力，限制注意力计算的可视范围
3、稀疏化，筛选有限的token做注意力计算

NSA在KV序列进行不同尺度的筛选：压缩、选择和滑动窗口，对应全局信息、重要信息以及就近信息。
1、压缩得到全局信息，讲kv 分段，然后每段压缩（在token维度进行降维d*l -> d*c）,然后进行注意力计算，得到全局信息o_cmp
2、压缩时，得到了每段的注意力分数，通过这个注意力分数选取top-k个段落，进行选择注意力的计算，得到o_sel。
3、取就近的kv，计算滑窗注意力o_win
4、门控，将三种注意力进行汇聚：$$ \sum g @ o $$ 这里的门控通过输入x 进行线性层变化+激活函数得到

假设上下文长度为64k（65536），分成128块*512
  全局压缩kv:得到128个KV对
  选择kv：选择top-8个，得到512*8个KV对
  就近窗口kv：就近8个，得到512*8个KV对
相比原本的65536个，现在只有128+512*8*2=8320个了，相当于一个有设计的降维。

### 3、DSA
DSA是deepseek V3.2中使用的技术，其并非像NSA那样是原生预训练的，V3.2是基于V3.1-T(Terminus)继续预训练，
random --pretrain--> NSA
random --pretrain--> MLA --continue pretrain--> MLA+DSA
DSA更多的考虑了精度，KVcache管理，MLA适配和Kernel设计等工程问题，NSA考虑的是blockwise的，但DSA直接使用索引来筛选element wiseKV 更加的灵活，不再是一块一块的了

DSA 的设计改进包括 MLA + indexr + MQA
1、MLA的 $$ c_t^{KV} $$是单头的
2、indexer: 索引网络存在W_q 和 W_k 投影矩阵，通过运算并取top-k 得到稀疏的KV
3、因为 $$ c_t^{KV} $$是单头的，所以使用MQA进行运算





### 4、比较

相比较而言，MoBA的做法更加直观，将MoE的思想运用到了attention上面，而且实验做到了10M token，尝试了不同的组合（全注意力和MoBA在实验阶段和不同层级），1M实现6.5倍速度提升，10M实现16倍提升。
