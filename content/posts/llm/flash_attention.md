---
title: 'Flash Attention'
date: 2024-05-23T14:02:18Z
lastmod: 2024-05-23
draft: false
tags: ["NLP", "LLM", "Attention"]
series: ["Attention"]
series_order: 1
# layout: "simple"
---

FlashAttention: [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

与标准 attention 相比，Flash Attention 有以下三点特点：
- 运算速度更快 (Fast)
- 更节省显存 (Memory-Efficient)
- 计算结果相同 (Exact)

> FlashAttention 目的不是节约 FLOPs，而是减少对HBM的访问。它没有改变原有的计算公式，整体计算复杂度并未降低。

## 背景

GPU中存储单元主要有 HBM 和 SRAM：HBM 容量大但是访问速度慢，SRAM容量小却有着较高的访问速度。例如：A100 GPU有40-80GB的HBM，带宽为1.5-2.0TB/s；每108个流式多核处理器各有192KB的片上SRAM，带宽估计约为 19TB/s。可以看出，片上的SRAM比HBM快一个数量级，但尺寸要小许多数量级。

当输入序列（sequence length）较长时，Transformer的计算过程缓慢且耗费内存，这是因为 self-attention 的 time 和 memory complexity 会随着 sequence length 的增加成二次增长。

标准Attention的计算过程：
{{< katex >}}
$$
S=Q K^T \in \mathbb{R}^{N \times N}
$$
$$
P=\operatorname{softmax}(S) \in \mathbb{R}^{N \times N}
$$
$$
O=P V \in \mathbb{R}^{N \times N}
$$

标准Attention的中间结果 𝑆, 𝑃 通常需要通过高带宽内存（HBM）进行存取，两者所需内存空间复杂度为\\(O(Nd+N^2)\), 对 HBM 的重复读写是主要瓶颈。要解决这个问题，需要做两件事：

- 在不访问整个输入的情况下计算 softmax
- 不为反向传播存储大的中间 attention 矩阵(\\(N^2\\))


## FlashAttention V1

FlashAttention 提出了两种方法来解决上述问题：tiling 和 recomputation。

- tiling - **注意力计算被重新构造**，将输入分割成块，并通过在输入块上进行多次传递来递增地执行softmax操作。
- recomputation - 存储来自前向的 softmax 归一化因子，以便在反向中快速重新计算芯片上的 attention，这比从HBM读取中间矩阵的标准注意力方法更快。可以把它看作基于 tiling 的特殊的 gradient checkpointing

正常的softmax计算：

$$m(x):=\max _i x i$$
$$f(x):=\left[e^{x_1-m(x)} \ldots e^{x_B-m(x)}\right]$$
$$\ell(x):=\sum_i f(x)_i$$
$$\operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}$$

softmax 伪代码:
![softmax](https://pic4.zhimg.com/80/v2-b5b221b2a8ef9b3602adef912668ea27_1440w.webp)

softmax 函数需要三个循环，第一个循环计算数组的最大值，第二个循环计算 softmax 的分母，第三个循环计算 softmax 输出。

分块的 softmax 计算(假设分2块并行计算)：

$$m(x)=m\left(\left[x^{(1)} x^{(2)}\right]\right)=\max \left(m\left(x^{(1)}\right), m\left(x^{(2)}\right)\right) $$
$$f(x)=\left[e^{m\left(x^{(1)}\right)-m(x)} f\left(x^{(1)}\right) \quad e^{m\left(x^{(2)}\right)-m(x)} f\left(x^{(2)}\right)\right] $$
$$\ell(x)=\ell\left(\left[x^{(1)} x^{(2)}\right]\right)=e^{m\left(x^{(1)}\right)-m(x)} \ell\left(x^{(1)}\right)+e^{m\left(x^{(2)}\right)-m(x)} \ell\left(x^{(2)}\right) $$
$$\operatorname{softmax}(x)=\frac{f(x)}{\ell(x)}$$

分块的 softmax 伪代码:
![tilling softmax](https://pic2.zhimg.com/80/v2-97d9313fbc337b46c171bc722dcafdbd_1440w.webp)

在第一个循环中同时对最大值\\(m\\)以及 softmax 的分母\\(d\\)进行更新，从而减少了一个循环。通过 tiling 的方式，softmax 的循环数从三个减到了两个，从而可以降低内存消耗。

flashattention 伪代码：
![flashattention](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F8ed46b76-4667-4e7d-a1e8-9c10de04c82a%2FUntitled.png?table=block&id=af426072-791a-449d-86e7-8ccb82240c17&t=af426072-791a-449d-86e7-8ccb82240c17)

中间变量：\\(O_i\\)(最终乘积)、\\(l_i\\)（softmax的分母，即累加和）、\\(m_i\\)（遍历到当前块为止的最大值），再也不用保存全部的S和P了。

> 由于重新计算导致FLOPs增加，但是由于大量减少HBM访问，FlashAttention运行速度更快

FlashAttention的 FLOPs 为\\(𝑂(𝑁^2𝑑)\\)，除了input和output，额外需要的内存为\\(𝑂(𝑁)\\), 对HBM访问的次数为\\(𝑂(𝑁^2𝑑^2𝑀^{−1})\\), 比标准 Attention 的\\(O(Nd+N^2)\)更高效

> PyTorch 2.0已将 FlashAttention 集成到官方库中，可以直接调用[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)


## 总结

FlashAttention V1:
- 通过切块技术减少了内存访问次数，提高了计算速度和内存利用率。
- 内存访问复杂度为\\(𝑂(𝑁^2𝑑^2𝑀^{−1})\\), 比标准 Attention 的\\(O(Nd+N^2)\)更高效