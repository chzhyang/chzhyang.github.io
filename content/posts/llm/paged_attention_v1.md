---
title: "Paged Attention V1(vLLM)"
date: 2024-05-20T15:15:49Z
lastmod: 2024-05-24
draft: false
description: ""
tags: ["NLP", "Transformer", "LLM", "vLLM", "Paged Attention"]
series: ["vLLM", "Attention and Optimization"]
series_order: [1, 1]
# layout: "simple"
showDate: true
---

## vLLM

vLLM是吞吐性能卓越的大模型推理框架，PagedAttention是vLLM最大的创新点： [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/abs/10.1145/3600006.3613165)


vLLM中的attention计算，在推理的prefill阶段, 使用第三方库xformers的优化实现，decoding阶段使用 CUDA kernel 实现(csrc/attention/attention_kernels.cu，大约800多行)。

Attention计算时使用页式管理 KV Cache 来提高内存利用率，进而提高吞吐量。

## Paged Attention(PA)

vLLM中有两个版本的 PA，其中：
- V1 源于 FasterTransformers 的 MHA，适用于 len(seq) < 8192 或 num_seqs * num_heads > 512 的情况。
- V2 参考了 Flash Decoding方式，对 sequence 的维度进行切分来增加并行粒度


## Paged Attention V1 and CUDA Kernel

[csrc/attention/attention_kernels.cu](https://github1s.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu#L90-L91)

并行任务的划分：
- dim3 grid(num_heads, num_seqs)
- dim3 block(NUM_THREADS), 线程数是128，每个 block 负责完成 output 矩阵一行（head_size个元素）结果的 attention 计算
- block 的线程划分为若干个 Warp, 每个 Warp 的32个线程划分为 blk_size 个 thread group

Kernel 输入参数
```python
out[num_seqs, num_heads, head_size]
q[num_seqs, num_heads, head_size]
k_cache[num_blocks, num_kv_heads, head_size/x, block_size, x] # x表示一个向量化的大小，如float16 -> 16 / sizeof(float16) = 8
v_cache[num_blocks, num_kv_heads, head_size, block_size]
head_mapping[num_heads] # 使用MQA, GQA时的kv_head
block_tables[num_seqs, max_num_blocks_per_seq] # 维护各个Q对应KVCache的哪些block
context_lens[num_seqs] # 用于变长
```
num_head： Q 的 head 数
num_kv_heads：K, V 的 head 数，MHA 的 num_kv_heads = num_head，GQA、MQA 的 num_kv_heads < num_head
blk_size # block_size，每个page block存储的元素数量，每个page存(blk_size, num_head，head_size)个K、V的元素

Kernel 的常量定义：
- THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1) 通过WARPSIZE / BLOCKSIZE 得到一个thread_group大小。注意这里的BLOCKSIZE不是cuda blocksize，而是一个kv block的大小(默认值16)
- NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / - WARP_SIZE 表示每个thread_group处理多少个token
- NUM_WARPS 表示一个threadblock有多少个warp
- VEC_SIZE 表示向量化大小，保证每个thread_group一次性获取16bytes，MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1)
- NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE 表示每个thread要负责多少个数据计算
- NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE 表示每个thread负责的数据经过向量化后，一共有多少个vec
- V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE) 每个thread一次性读取16bytes
- NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE。对于v_cache[head_size, block_size]，表示一行需要几个V_VEC
- NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW 表示一个warp可以处理多少行
- NUM_ROWS_PER_THREAD 表示每个thread需要负责多少行


{{< katex >}}

从显存读取\\(Q\\)到 shared memory：

迭代读取，每 CUDA block 负责读取\\(Q\\)的一行（head_size 个元素）存入 shared memory。其中，block 的每个 Warp 负责读取 16*blk_size 字节的 Q，即每个 thread group 会读取16字节的 Q，16*blk_size 字节的 Q 对应 sequence 的一个 head。
```c++
const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

// Load the query to registers.
// Each thread in a thread group has a different part of the query.
// For example, if the the thread group size is 4, then the first thread in
// the group has 0, 4, 8, ... th vectors of the query, and the second thread
// has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
// q is split from a qkv tensor, it may not be contiguous.
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
    i += NUM_THREAD_GROUPS) {
const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
q_vecs[thread_group_offset][i] =
    *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
}
__syncthreads();
```

从显存读取\\(K\\)到 register：

- 每个 seq 包含 cxt_length * num_kv_heads * head_size 个元素
- 每个 CUDA block 负责计算一个 seq 的一个 head 的 \\(QK^T\\)， 只需要读取 ctx_length * head_size 个 K 的元素
- 因为页式内存管理，K 在 ctx_length 维度的存储不连续，以 blk_size 个 token 为粒度分布在不同的内存地址，所以需要根据 Q 的 head_idx 和 seq_idx 访问 block_table 找到 K 的 physical_block_num
- K Cache的布局为 [num_blocks, num_kv_heads, head_size/x, block_size, x]， 目的是优化写入 shared memory。Q和K的同一行元素被读入寄存器并进行点乘运算后，结果要写入shared memory。如果一个 Warp 中所有线程都计算 Q、K 同一行数据，会导致写入 shared memory 的同一个位置，这将造成 warp 内不同线程顺序地写入。所以 warp 的线程最好计算 Q和K 的不同行数据。在设计 K 布局时，将 block_size 放在比 head_size 更低的维度。由于warp size大于block_size，我们需要将head_size拆分为head_size/x和x两个维度，借x到最低维度，以确保每个线程读入的数据量和计算量都足够大。最后，每个线程组派一个线程去写入shared memory，这样一个warp有blk_size个线程并行写入shared memory，从而增加了shared memory的访问带宽。这种设计策略是为了实现高效的并行计算和内存访问，以提高整体的计算性能。
- 读取 K 需要一个循环，循环中每个CUDA block中的所有 warp 依次访问num_blocks 个 page block。每次迭代：
    - 每个 warp 负责访问连续的 blk_size 个 KCache 的行数据（blk_size * head_size个元素）。每个 thread group 负责访问 KCache 的一行，将head_size 个元素读入寄存器
    - 寄存器中的Q和K元素进行点乘，结果写入shared memory。一个 CUDA block 的 shared memory 存储了一行 QK^T 的结果，共 ctx_length 个元素
    - CUDA block 对 shared memory 中元素进行 max，sum 方式 reduction，然后计算得到 softmax 的结果

从显存读取\\(K\\)到 register：

和K Cache一样，CUDA thread block依次访问num_blk个物理块到寄存器，每个warp负责blk_size个token的page内存，page的真实物理地址同样需要进行索引。不过这里不需要以thread group为单位访问16字节，而是每个thread访问16字节的元素。访问完就可以与shared memory的softmax(QK^T)中间结果对应位置16字节的数据进行点乘，得到一个float结果，写到output对应位置中。

> 为什么 VCache 的 layout 是 [num_blocks, num_kv_heads, head_size, block_size]，和 KCache layout 不一样？ 因为 V 要去做点乘的对象在shared memory，只需要读，不涉及并行写。


## PA V1 和 Flash Attention 的区别

并行任务的划分方式不同
- FlashAttention 用了两层循环，每次写一个 Tile 的 output tensor，而 PA 只有一层循环，每次写一行 output tensor。因为每次迭代都有整行的 QK^T 中间结果，不需要online softmax
- PA V1 设计的 KCache layout 充分利用了 shared memory 写带宽

## PA V1 的缺陷

不足：
- 不适合 seq 很长的情况，因为没有沿着 ctx_length 或者 batch 维度做切分
- 和MHA相比，MQA和GAQ没有减少对KV Cache的读写次数。读K、V Cache时候只是做了一个head_idx的转换，会重复从显存读相同的head

未完待续...

Reference:
- [vllm](https://www.zhihu.com/question/633412311/answer/3332907958)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/abs/10.1145/3600006.3613165)