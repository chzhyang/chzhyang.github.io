---
title: "CUDA Memory and Optimization"
date: 2024-05-23T13:57:32Z
draft: false
description: ""
tags: ["CUDA"]
series: ["CUDA Parallel Programming"]
series_order: 5
# layout: "simple"
---

## Global Memory

全局内存的访问模式，有合并（coalesced）与非合并（uncoalesced）之分。

合并访问指的是一个线程束对全局内存的一次访问请求（读或者写）导致最少数量的数据传输，否则称访问是非合并的。

合并度（degree of coalescing）等于线程束请求的字节数除以由该请求导致的所有数据传输处理的字节数。如果所有数据传输中处理的数据都是线程束所需要的，那么合并度就是 100%，即对应合并访问。也可以将合并度理解为一种资源利用率。利用率越高，核函数中与全局内存访问有关的部分的性能就更好

顺序的合并访问:

```c++
void __global__ add(float *x, float *y, float *z)
{
int n = threadIdx.x + blockIdx.x * blockDim.x;
z[n] = x[n] + y[n];
}
add<<<128, 32>>>(x, y, z);
```
第一个线程块中的线程束将访问数组 x 中第 0-31 个元素，对应 128 字节的连续内存，而且首地址一定是 256 字节的整数倍。这样的访问只需要 4 次数据传输即可完成，所以是合并访问，合并度为 100%

乱序的合并访问:

```c++
void __global__ add_permuted(float *x, float *y, float *z)
{
int tid_permuted = threadIdx.x ^ 0x1;//是将 0-31 的整数做某种置换（交换两个相邻的数）
int n = tid_permuted + blockIdx.x * blockDim.x;
z[n] = x[n] + y[n];
}
add_permuted<<<128, 32>>>(x, y, z);
```
第一个线程块中的线程束将依然访问数组 x 中第 0-31 个元素，只不过线程号与数组元素指标不完全一致而已，合并度也为 100%

不对齐的非合并访问:

```c++
void __global__ add_offset(float *x, float *y, float *z)
{
int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
z[n] = x[n] + y[n];
}
add_offset<<<128, 32>>>(x, y, z);
```
第一个线程块中的线程束将访问数组 x 中第 1-32 个元素。假如数组 x 的首地址为 256字节，该线程束将访问设备内存的 260-387 字节。这将触发 5 次数据传输，对应的内存地址分别是256-287 字节、288-319 字节、320-351 字节、352-383 字节和 384-415 字节，合并度为 4/5 = 80%

跨越式的非合并访问

```
void __global__ add_stride(float *x, float *y, float *z)
{
int n = blockIdx.x + threadIdx.x * gridDim.x;
z[n] = x[n] + y[n];
}
add_stride<<<128, 32>>>(x, y, z);
```
第一个线程块中的线程束将访问数组 x 中指标为 0、128、256、384 等的元素。每一对数据都不在一个连续的 32 字节的内存片段，故该线程束的访问将触发 32 次数据传输，合并度为 4/32 = 12.5%

### CUDA Kernel - Matrix Transpose

对于多维数组，x 维度的线程指标 threadIdx.x 是最内层的（变化最快），所以相邻的 threadIdx.x 对应相邻的线程，即对threadIdx.x相邻的数据的访问是连续的。

CUDA中，`顺序读`的性能理论上高于`非顺序读`，但是实际性能一致。因为从帕斯卡架构开始，如果编译器能够判断一个全局内存变量在整个核函数的范围都只可读（如这里的矩阵 A），则会自动用函数 __ldg() 读取全局内存，从而对数据的读取进行缓存，缓解非合并访问带来的影响。

但是写操作没有这种自动配置，所以 `顺序写`的实际性能高于`非顺序写`。**所以，CUDA中访问全局内存时，要注意优先做到顺序写。**

```c++
__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx]; //顺序读，非顺序写
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];//顺序写，非顺序读
    }
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);//顺序写，自动化
    }
}
```

并行配置：

```c++
int N=1024;
int grid_x = (N+32)/N); //32 = float32的大小，一个thread访问一个float32数据
dim3 grid(grid_x, grid_x);
dim3 block(32,32)//block最多1024个threads
int M = sizeof(real) * N2;
real *d_A, *d_B;
cudaMallocManaged(&d_A, M);
cudaMallocManaged(&d_B, M);
```

在Tesla T4上测试结果：

```
transpose with coalesced read:
Time = 0.193677 +- 0.000643045 ms.

transpose with coalesced write:
Time = 0.129763 +- 0.00107203 ms.

transpose with coalesced write and __ldg read:
Time = 0.130074 +- 0.00122755 ms.

```

## Shared memory

上面矩阵转置例子中，对全局内存的读和写这两个操作，总有一个是合并的，另—个是非合并的。利用共享内存可以改善全局内存的访问模式，使得对全局内存的读和写都是合并的。

核心思路是用一个 block 处理 BLOCK_SIZE * BLOCK_SIZE 的矩阵块

```c++
const int block_size = 32;
__global__ void matrix_trans(int* in, int* out){
    __shared__ int buf[block_size];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n = in.size();
    if(i<n && j<n){
        buf[threadIdx.y][threadIdx.x] = in[i*n+j];
    }
    __syncthreads();

    i = blockIdx.y * blockDim.y + threadIdx.x;
    j = blockIdx.x * blockDim.x + threadIdx.y;
    if(i<n&&j<n>){
        out[j*n+i] = buf[threadIdx.x][threadIdx.y];
    }
}

```

### CUDA Kernel - Array Reduce

{{< katex >}}

一个有 N（\\(10^8\\)） 个元素的数组 x，假如我们需要计算该数组中所有元素的和，即 sum = x[0] + x[1] + ... + x[N - 1]。

Todo: CUDA编程：基础与实践 P83

Reference:

- [CUDA编程：基础与实践](https://book.douban.com/subject/35252459/)