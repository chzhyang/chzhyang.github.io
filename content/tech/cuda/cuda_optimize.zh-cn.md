---
title: "CUDA Optimization"
date: 2024-05-24T13:57:32Z
draft: false
description: ""
tags: ["CUDA"]
series: ["CUDA Parallel Programming"]
series_order: 4
# layout: "simple"
---

CUDA 程序获得高性能的必要（但不充分）条件有：
- 数据传输比例较小
- 核函数的算术强度较高（计算访存比）
- 核函数中定义的线程数目较多

在编写与优化 CUDA 程序时，要想方设法（设计算法）做到：
- 减少主机与设备之间的数据传输
- 提高核函数的算术强度（计算访存比）
- 增大核函数的并行规模