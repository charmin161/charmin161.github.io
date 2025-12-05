在 CUDA 编程中，智能指针（Smart Pointers）通常不会出现在 Kernel（__global__ 函数）内部，因为 GPU 无法像 CPU 那样处理复杂的对象生命周期和原子计数。

它们真正的用武之地在于 Host 端（CPU）的代码，用来管理 Device 端（GPU）的资源。在大型项目中，智能指针主要解决两个核心痛点:"
- 1、 资源泄露（RAII）：忘记 cudaFree 或 cublasDestroy。
- 2、零拷贝共享：在不同算子或层之间传递 Tensor 数据，而不触发昂贵的 GPU 内存拷贝。
