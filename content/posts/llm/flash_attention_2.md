---
title: "Flash Attention V2"
date: 2024-05-23T22:43:31Z
draft: false
description: ""
tags: ["NLP", "Transformer", "LLM", "Attention"]
series: ["Attention"]
series_order: 2
# layout: "simple"
showDate: true
---

[WIP]

FlashAttention V2:
- 在V1的基础上减少了非矩阵乘法运算的FLOPs。
- 通过并行化和任务分配优化提高了计算速度和GPU利用率，性能提升了2-3倍。
- Flash-Decoding借鉴了FlashAttention的优点，将并行化维度扩展到keys/values序列长度，提高了推理速度。
- Flash-Decoding几乎不用额外存储大量数据到全局内存中，减少了内存开销。
- Flash-Decoding++通过异步softmax和统一最大值、flat GEMM优化和双缓冲、启发式数据流和硬件资源适应等方法进一步提高了LLM推理的性能。