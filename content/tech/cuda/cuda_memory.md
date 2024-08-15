---
title: "CUDA Memory and Optimization"
date: 2024-05-23T13:57:32Z
lastmod: 2024-05-29
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
int TILE_DIM = 32;//32 = float32的大小，一个thread访问一个float32数据
int grid_x = (N+TILE_DIM-1)/TILE_DIM); 
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

上面的矩阵转置例子中，对全局内存的读和写这两个操作，总有一个是合并的，另—个是非合并的。**利用共享内存可以改善全局内存的访问模式，使得对全局内存的读和写都是合并的。**

全局内存的访问速度是所有内存中最低的，应该尽量减少对它的使用。所有设备内存中，寄存器是最高效的，但在需要线程合作的问题中，用仅对单个线程可见的寄存器是不够的, 需要使用对整个线程块可见的共享内存：

- 在核函数中，要将一个变量定义为共享内存变量，就要在定义语句中加上一个限定符 `__shared__`。一般情况下，共享内存的数组长度等于线程块大小
- 线程块的处理逻辑完成后，在利用共享内存进行线程块之间的合作（通信）之前，都要进行同步 `__syncthreads()`，以确保共享内存变量中的数据对线程块内的所有线程来说都准备就绪
- 因为共享内存变量的生命周期仅仅在核函数内，所以必须在核函数结束之前将共享内存中的某些结果保存到全局内存

动态共享内存和静态共享内存：
- 静态的限定符 `__shared__`， 需要指定内存大小，如 `__shared__ real s_y[128]`
- 动态的限定符 `__extern__`, 定义变量时不用指定内存大小，如 `extern __shared__ real s_y[]`， 但是需要在调用 kernel 时，加入共享内存的参数，即**共享内存的数组长度等于线程块大小**，如 `kernel<<<grid_size, block_size, sizeof(real) * block_size>>>()`, 这个效果与静态共享内存是一样的，但可以防止使用静态共享内存时指定内存长度时出错
- 使用动态共享内存的核函数和使用静态共享内存的核函数在执行时间上几乎没有差别。但使用动态共享内存容易可提高程序的可维护性

上面的矩阵转置例子使用共享内存的思路是用一个 block 处理 BLOCK_SIZE * BLOCK_SIZE 的矩阵块

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

- 先调用 kernel 将数组 x 归约到 grid_size 大小，即每个线程块完成 block_size 大小的归约，结果写到数组 y (y 的长度为 grid_size)
- 然后在 host 上 完成最后一步的归约，即 y[0...grid_size-1] -> result

Kernel如下：
```c++
// 全局内存
void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}
// 静态共享内存
void __global__ reduce_shared(real *d_x, real *d_y){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n<N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset=blockDim.x >> 1; offset >0; offset >>= 1){
        if(tid<offset>){
            s_y[tid]=s_y[tid+offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}
//动态共享内存
void __global__ reduce_dynamic(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}
```

Host上完成最后一步：

```c++
real result = 0.0;
for (int n = 0; n < grid_size; ++n)
{
    result += h_y[n];
}
```

Telsa T4上的性能对比，详细参数见 [code](https://github.com/MAhaitao999/CUDA_Programming/blob/master/CUDA/chapter8_%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E5%90%88%E7%90%86%E4%BD%BF%E7%94%A8/reduce2gpu.cu)：

```
Using global memory only:
Time sum = 123633392.000000.

Using static shared memory:
Time sum = 123633392.000000.

Using dynamic shared memory:
Time sum = 123633392.000000.
```
### Bank Conflict in Shared Memory

有一个内存 bank 的概念值得注意。为了获得高的内存带宽，共享内存在物理上被分为 32 个（刚好等于一个线程束中的线程数目，即内建变量 warpSize 的值）同样宽度的、能被同时访问的内存 bank。

将 32 个 bank 从 0 到 31 编号。在每一个 bank 中，又可以对其中的内存地址从 0 开始编号。所有 bank 中编号为 0 的内存称为第一层内存；将所有 bank 中编号为 1 的内存称为第二层内存。

对于 bank 宽度为 4 字节的架构(开普勒)，共享内存数组是按如下方式线性地映射到内存 bank 的：共享内存数组中连续的 128 字节的内容分摊到 32 个 bank 的某一层中，每个 bank 负责 4 字节的内容。例如：对一个长度为 128 的单精度浮点数变量的共享内存数组来说，每个 bank 分摊 4 个在地址上相差 128 字节的数据：
- 第 0-31 个数组元素依次对应到 32 个 bank 的第一层
- 第 32-63 个数组元素依次对应到 32 个 bank 的第二层
- 第 64-95 个数组元素依次对应到 32 个 bank 的第三层
- 第 96-127 个数组元素依次对应到 32 个 bank 的第四层

只要同一线程束内的多个线程不同时访问同一个 bank 中不同层的数据，该线程束对共享内存的访问就只需要一次内存事务（memory transaction）。当同一线程束内的多个线程试图访问同一个 bank 中不同层的数据时，就会发生 bank 冲突。在一个线程束内对同一个 bank 中的 n 层数据同时访问将导致 n 次内存事务，称为发生了 n 路 bank 冲突。最坏的情况是线程束内的 32 个线程同时访问同一个 bank 中 32 个不同层的地址，这将导致 32 路 bank 冲突。这种 n 很大的 bank 冲突是要尽量避免的。

如何消除冲突？ 可以用改变共享内存数组大小的方式来消除或减轻共享内存的 bank 冲突。详细方法见下面的矩阵转置例子transpose2。

### CUDA Kernel - Matrix Transpose(Shared Memory)

核心思想：
- 用一个线程块处理一块(tile)的矩阵，比如设置 tile 的边长 TILE_DIM = 32
- 将一个 tile 的矩阵从全局内存数组 A 中读入线程块的共享内存（二维数组 S[TILE_DIM][TILE_DIM], 线程也是按照二维的方式去执行）
- 线程块迭代归约整个 tile（）

Kernel:
```c++
// N 为矩阵的边长
__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;

    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

// 消除bank conflict
__global__ void transpose2(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

```

```c++
const int N = 1024;
const int TILE_DIM = 32;// 线程块 要处理的二维数据的维度为（TILE_DIM，TILE_DIM）
const dim3 block_size(TILE_DIM,TILE_DIM);
const int grid_size_x = (N + TILE_DIM-1)/N;
const dim3 grid_size(grid_size_x,grid_size_x);

const int M = sizeof(real) * (N*N);
real *d_A = (real *) malloc(M);//输入数组X
real *d_B = (real *) malloc(M);//结果数组Y，需要做最后一轮归约
// init A
for (int i = 0; i < N*N; ++i)
{
    h_A[i] = i;
}
cudaMallocManaged(&d_A, M);
cudaMallocManaged(&d_B, M);
transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
cudaFree(d_A);
cudaFree(d_B);
```
