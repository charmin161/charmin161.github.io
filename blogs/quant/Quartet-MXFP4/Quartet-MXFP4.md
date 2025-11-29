# Quartet:MXFP4 量化方法

1、使用了一种基于量化训练过程中的scaling law的评价准则

![](https://charmin161.github.io/blogs/quant/Quartet-MXFP4/image/image_qFov33n5xH.png)

其中 $eff_N(P_{forward})$ 是基于前向与参数相关的系数； $eff_D(P_{baskward})$ 是基于后向与数据相关的系数，这两个的取值范围都是（0，1]。

前向时，需要量化后的模型精度（MSE）尽可能与原模型一致，考虑参数效率和推理速度；反向时，需要尽快收敛，考虑数据效率和训练速度，不要让梯度无偏（梯度无偏对收敛影响很大）\[1]

无偏量化：

3.3 = 0.7\*3+0.3\*4

舍入的结果在统计意义上与原值相同。方向（direction）和幅度(migntitude）上的无偏，将整个张量视为一个向量，其方向和范数上的无偏。

2、基于这套准则进行实验，选择了MXFP4的最优量化方案：

![](https://charmin161.github.io/blogs/quant/Quartet-MXFP4/image/image_Ch5vA8JLjU.png)

1）前向使用Hadamard变换，相比AbsMax的归一化效果更好（ $eff_N$ 更大）

2）使用QuEST，基于均方根误差的裁剪

3\)  反向使用随机Hadamard变换和随机舍入，目的是无偏量化

\[1] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communicationefficient sgd via gradient quantization and encoding. Advances in neural information processing systems,
30, 2017.
