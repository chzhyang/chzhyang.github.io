---
title: "GPU 结构"
date: 2024-05-18T09:47:15Z
draft: false
description: ""
tags: ["GPU", "CUDA"]
series: ["CUDA Parallel Programming"]
series_order: 1
# layout: "simple"
showDate: true
---

## CPU 与 GPU 的不同

![diff_gpu_cpu](diff_gpu_cpu.png)

- CPU，4个 ALU，主要负责逻辑计算，1个控制单元 Control，1个 DRAM，1个 Cache
- GPU，绿色小方块看作 ALU，红色框看作一个 SM，SM 中的多个 ALU share 一个Control 和 Cache，SM 可以看作一个多核 CPU，但是 ALU 更多，control 更少，也就是算力提升，控制力减弱

所以，CPU 适合控制逻辑复杂的任务，GPU 适合逻辑简单、数据量大、计算量大的任务。

## GPU, CUDA, AI Framework 的关系

![relationshape](https://img-blog.csdnimg.cn/5d8d7665a06942fe95195257ae3a80e5.png)

Reference:

- [NVIDIA CUDA Docs](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [cuda编程学习](https://blog.csdn.net/qq_40514113/article/details/130818169?spm=1001.2014.3001.5502)
- [一张图了解GPU、CUDA、CUDA toolkit和pytorch的关系](https://blog.csdn.net/Blackoutdragon/article/details/130233562)
- [GPU 内存概念浅析](https://zhuanlan.zhihu.com/p/651179378)

## GPU 内部结构

每一个 SM 有自己的 Wrap scheduler 、寄存器（Register）、指令缓存、L1缓存、共享内存。

A100 中每个 SM 包括 4 个 SM partition（SMP），里边绿色的就是 Streaming Processor（SP），也叫 CUDA cores，它们是实际执行计算的基本单元。

![gpu arch](https://img2024.cnblogs.com/blog/798398/202402/798398-20240219182315457-1097510645.png)

所有的 SM 共享 L2 缓存。整个 GPU 内存结构如下图所示

![memroy share](https://img2024.cnblogs.com/blog/798398/202402/798398-20240219182333769-1289077570.png)

## GPU 内存结构

按照存储功能进行细分，GPU 内存可以分为：局部内存（local memory）、全局内存（global memory）、常量内存（constant memory）、共享内存（shared memory）、寄存器（register）、L1/L2 缓存等。

其中全局内存、局部内存、常量内存都是片下内存(off-chip)，储存在 HBM 上。所以 HBM 的大部分作为全局内存。

- on-chip：L1/L2 cache：多级缓存，在 GPU 芯片内部
- off-chip：GPU DRAM/HBM, global memory

- L2 缓存可以被所有 SM 访问，速度比全局内存快。Flash attention 的思路就是尽可能地利用 L2 缓存，减少 HBM 的数据读写时间
- L1 缓存用于存储 SM 内的数据，被 SM 内的 CUDA cores 共享，但是跨 SM 之间的 L1 不能相互访问
- 局部内存 (local memory) 是线程独享的内存资源，线程之间不可以相互访问。局部内存属于off-chip，所以访问速度跟全局内存一样。它主要是用来应对寄存器不足时的场景，即在线程申请的变量超过可用的寄存器大小时，nvcc 会自动将一部数据放置到片下内存里。
- 寄存器（register）是线程能独立访问的资源，它是片上（on chip）存储，用来存储一些线程的暂存数据。寄存器的速度是访问中最快的，但是它的容量较小，只有几百甚至几十 KB，而且要被许多线程均分
- 共享内存（shared memory） 是一种在线程块内能访问的内存，是片上（on chip）存储，访问速度较快。共享内存主要是缓存一些需要反复读写的数据。共享内存与 L1 缓存的位置、速度极其类似，区别在于共享内存的控制与生命周期管理与 L1 不同：共享内存受用户控制，L1 受系统控制。共享内存更利于线程块之间数据交互。
- 常量内存（constant memory）是片下（off chip）存储，但是通过特殊的常量内存缓存（constant cache）进行缓存读取，它是只读内存。常量内存主要是解决一个 warp scheduler 内多个线程访问相同数据时速度太慢的问题。假设所有线程都需要访问一个 constant_A 的常量，在存储介质上 constant_A 的数据只保存了一份，而内存的物理读取方式决定了多个线程不能在同一时刻读取到该变量，所以会出现先后访问的问题，这样使得并行计算的线程出现了运算时差。常量内存正是解决这样的问题而设置的，它有对应的 cache 位置产生多个副本，让线程访问时不存在冲突，从而保证并行度。

## Tensor Core

CUDA core 和 Tensor core 的区别：
- Tensor core 是在 Volta 以及之后的架构中才有的, 相比于CUDA core，它可以提供更高效的运算。
- 每个 GPU clock，CUDA core 可以进行一次单精度乘加运算，即：in fp32: x += y * z。
- 每个 GPU clock，Tensor core 可以完成 4 × 4 的混合精度矩阵乘加 (matrix multiply-accumulate, MMA)：D=A * B + C，其中 A、B、C、D 都是 4 × 4 矩阵。A 和 B是 FP16 矩阵，而累加矩阵 C 和 D 可以是 FP16 或 FP32 矩阵（FP16/FP16 或 FP16/FP32 两种模式。所以每个 GPU clock，Tensor core 可以执行 64 个浮点 FMA 混合精度运算（4 × 4 × 4）。
- Turing 架构中新增了 INT8/INT32, INT4/INT32, INT1/INT32 等模式

![Tensor Core](https://img2024.cnblogs.com/blog/798398/202402/798398-20240219182403297-314074416.png)

V100 中，一个 SM 中有 8 个 Tensor core，每个 GPU clock 共可以执行 1024 个浮点运算（64 × 8 × 2，乘以 2 因为乘加是两个浮点运算）

Reference:
- [TENSOR CORE DL PERFORMANCE GUIDE](https://link.zhihu.com/?target=https%3A//developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf)
- [GPU内存概念浅析](https://www.cnblogs.com/ArsenalfanInECNU/p/18021724)