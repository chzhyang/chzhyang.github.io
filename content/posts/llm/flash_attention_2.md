---
title: "Flash Attention V2"
date: 2024-05-23T22:43:31Z
lastmod: 2024-06-03
draft: false
description: ""
tags: ["NLP", "Transformer", "LLM", "Attention"]
series: ["Attention and Optimization"]
series_order: 3
# layout: "simple"
showDate: true
---

{{<katex>}}

## FlashAttention-V2

FlashAttention2 对FlashAttention 的改进:
- 减少了**非矩阵乘法运算**
- 循环调整

## 减少非矩阵运算

A100的 FP16/BF16 矩阵乘的最大理论吞吐量为 312 TFLOPs/s，但FP32 非矩阵乘法仅有 19.5 TFLOPs/s，即每个 no-matmul FLOP比mat-mul FLOP 昂贵16倍。

V2 再算法上相对 V1 做了如下调整：

V2 在做 Attention 计算局部数值的时候，暂时不考虑分母，因此在嵌套计算 $O_i$ 的时候就不用做除法缩放，而是在最后一把调整。

V1 计算局部数值的过程：

$\begin{aligned}
& m^{(1)}=\operatorname{rowmax}\left(\mathbf{S}^{(1)}\right) \in \mathbb{R}^{B_r} \\
& \ell^{(1)}=\operatorname{rowsum}\left(e^{\mathbf{S}^{(1)}-m^{(1)}}\right) \in \mathbb{R}^{B_r} \\
& \tilde{\mathbf{P}}^{(1)}=\operatorname{diag}\left(\ell^{(1)}\right)^{-1} e^{S^{(1)}-m^{(1)}} \in \mathbb{R}^{B_r \times B_c} \\
& \mathbf{O}^{(1)}=\tilde{\mathbf{P}}^{(1)} \mathbf{V}^{(1)}=\operatorname{diag}\left(\ell^{(1)}\right)^{-1} e^{\mathbf{S}^{(1)}-m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d} \\
&
\end{aligned}$

V2 计算局部数值的过程：

$\begin{aligned}
& m^{(1)}=\operatorname{rowmax}\left(\mathbf{S}^{(1)}\right) \in \mathbb{R}^{B_r} \\
& \ell^{(1)}=\operatorname{rowsum}\left(e^{\mathbf{S}^{(1)}-m^{(1)}}\right) \in \mathbb{R}^{B_r} \\
& \mathbf{O}^{(1)}=e^{\mathbf{S}^{(1)}-m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d} \\
&
\end{aligned}$


从而，V1 每次更新 $O$ 需要在上一轮 $O_{i-1}$ 的基础上进行 $diag(l(1)/l(2))$ 的缩放，V2 也不需要了，只需要在过程补偿max值：

$\tilde{\mathbf{O}}^{(2)}=\operatorname{diag}\left(e^{m^{(1)}-m^{(2)}}\right)^{-1} \tilde{\mathbf{O}}^{(1)}+e^{\mathbf{S}^{(2)}-m^{(2)}} \mathbf{V}^{(2)}=e^{s^{(1)}-m} \mathbf{V}^{(1)}+e^{s^{(2)}-m} \mathbf{V}^{(2)}$


## 循环调整

V1 会将 Q 按行切块，K和V 按列切块，然后进行双重的循环计算。那么谁在外循环，谁在内循环呢？

因为 O 其实是跟着行走的， 所以若以 Qi 为外循环（按行循环），其实不断地加载 Ki 和 Vi 进来运算，相关部分的 O 其实可以一把搞定。若以 Ki，Vi 为外循环（按列循环），加载 Q1...Qr 进来运算，则 O 要被不断地写入写出。

[![pktSJ2j.webp](https://s21.ax1x.com/2024/06/07/pktSJ2j.webp)](https://imgse.com/i/pktSJ2j)

上图(a)中 V1（列循环）将 K 和 V 按列切为4块，然后分给4个 warp 并行计算，且所有 warp 都可以访问 Q。warp 将 K 乘以 Q 得到部分 Si，然后 Si 经过局部softmax 后还需要乘以 V 的一部分得到 Oi。每次计算完 Qi 还会更新数据到 HBM(对上一次版本O先rescale再加上当前值)。这导致每个 warp 需要从 HBM 频繁读写Qi 以累加出总结果。这种方式被称为 `split-K`，非常低效，因为所有warp都需要从 HBM 频繁读写中间结果 (Oi,mi,li)。

上图(b)中 V2（行循环）将对 Q 的遍历移到到了外循环，K 和 V 移到了内循环，并将 Q 按行切到4个 warp，所有 warp 都可以访问 K 和 V。V1 每次内循环会导致 O的变化，之后通过写 HBM 更新 O。现在每次内循环处理的都是 O，此时 O 是存储在SRAM上，效率高于 V1。这样做最大的好处是可以把并行度从串行循环改成并行，拆分为 GPU 运算模型中的 Thread Block。


## Performance

在A100 80GB 上测量不同设置（有无 Causal mask、head size64 或 128）下的不同注意力机制的运行时间。 结果表明：
- V2 比 V1 和 xformers 中的FlashAttention 快2倍
- V2 在 Forward 时比 Triton 的 FlashAttention Kernel 快 1.3-1.5 倍，在 backward 时快 2 倍
- 与 PyTorch 中的标准kernel相比，V2的速度最高可提高 10 倍

[![pktSzFS.webp](https://s21.ax1x.com/2024/06/07/pktSzFS.webp)](https://imgse.com/i/pktSzFS)

## Summary

FlashAttention V2:
- 在V1的基础上减少了非矩阵乘法运算的FLOPs
- 通过并行化和任务分配优化提高了计算速度和GPU利用率，性能提升了2-3倍
- Flash-Decoding借鉴了FlashAttention的优点，将并行化维度扩展到keys/values序列长度，提高了推理速度
- Flash-Decoding几乎不用额外存储大量数据到全局内存中，减少了内存开销
- Flash-Decoding++通过异步softmax和统一最大值、flat GEMM优化和双缓冲、启发式数据流和硬件资源适应等方法进一步提高了LLM推理的性能

Reference
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [轻松读懂FlashAttention2](https://mp.weixin.qq.com/s?__biz=MzkzNDM4MDQyMg==&mid=2247485280&idx=1&sn=8cc173d8c3b88865c56f386e6b3683b3&chksm=c2bf5189f5c8d89f05914feb51ee228e5fbea009e10b756f7ee7086f33d8d1e7adccb15623e6&mpshare=1&scene=1&srcid=06074EnR1glvtb3wbbOhNhsu&sharer_shareinfo=cc784831ffcd9f6610aedb913ab3834f&sharer_shareinfo_first=cc784831ffcd9f6610aedb913ab3834f#rd)