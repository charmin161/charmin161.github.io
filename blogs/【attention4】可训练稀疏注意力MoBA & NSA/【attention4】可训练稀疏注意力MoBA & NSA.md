# 【attention4】可训练稀疏注意力MoBA & NSA

之前的文中介绍了MLA的降秩操作，巧妙地利用attention结构中的冗余性，通过降秩的方案，减少KV cache，加快推理速度。

这次的两篇文章，kimi推出的MoBA（Mixture of Block Attention）和deepseek推出的NSA（Native Sparse Attention），都利用了full attention中的冗余性，使用稀疏的注意力方案，可以说是把MLP那边的思想，在attention里又用了一遍。虽然知道冗余，但要**稀疏哪里**呢？在计算 $QK^T$的过程中，如果不算，就不知道哪里稀疏，但算了就没必要再稀疏了（没有减少计算量），顶多可以减少后面V的计算量。所以如果可以在训练过程中，**从数据中学到注意力的稀疏性**，就很妙了。这两篇文章的核心卖点就是这个。

本文首先简单介绍这两篇内容的主要思想，再提炼两篇内容的共同观点，并比较一下不同。

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

1）动态分层稀疏

2）以组为中心的数据加载

3）共享KV加载

4）网格循环调度

### 3、比较

相比较而言，MoBA的做法更加直观，将MoE的思想运用到了attention上面，而且实验做到了10M token，尝试了不同的组合（全注意力和MoBA在实验阶段和不同层级），1M实现6.5倍速度提升，10M实现16倍提升。
