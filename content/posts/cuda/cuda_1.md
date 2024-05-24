---
title: "CUDA Programming"
date: 2024-05-18T09:47:15Z
draft: false
description: ""
tags: ["CUDA"]
series: ["CUDA Parallel Programming"]
series_order: 2
# layout: "simple"
---

## Steps of CUDA Program

1. Config GPU(device), `cudaSetDevice()`
2. Allocate device memory, `cudaMalloc()`, `cudaMallocManaged()`
3. Allocate CPU(Host) memory
4. Copy data from host to device, `cudaMemcpy()`
5. Run kernel on device
6. Copy result from device to host, `cudaMemcpy()`
7. Print result on host
8. Release host and device memory, `cudaFree()`, `free()`

> CPU is always called host, the GPU is called device

C Example:

```C
// file: sample.cu
#include<stdint.h>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// This is a sample CUDA kernel, called on host and execte on device
__global__ void add(float* a)
{
	a[threadIdx.x] = 1;

}
int main(int argc, char** argv)
{
	// 1. Set device 
	cudaSetDevice(0);
	// 2. Allocate device memory
	float* dx;
	cudaMalloc((void**)&dx, 16 * sizeof(float));
	// 3. Allocate host memory
	float hx[16] = { 0 };
	// 4. Copy data from host to device
	cudaMemcpy(dx, hx, 16 * sizeof(float), cudaMemcpyHostToDevice);
    // 5. Run kernel on device
	add << <1, 16 >> > (dx);
	// 6. Copy result from device to host
	cudaMemcpy(hx, dx, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    // 7. Print result on host
	for (int i = 0; i < 16; i++)
	{
		printf("%f \n", hx[i]);
	}
	8. Release host and device memory
	cudaFree(dx);
	free(hx);
    //cudaDeviceReset();
	return 0;
}
```

使用`nvcc`编译,然后运行

```shell
nvcc sample.cu - o sample
./sample
```

> CUDA 提供了**统一内存**: gpu和cpu可访问的单一内存空间. 调用`cudaMallocManaged()`，它返回一个指针，从host code或device code都可以访问。要释放数据，只需将指针传递给cudaFree()。

## CUDA Kernel and Parrallel Computing

> 前置知识: 理解 GPU 结构, Grid, Block, Thread 这几个逻辑概念之间的关系


**CUDA kernel 的编程模型**

[Todo] Dim and size detail

调用kernel: `add << <blockNumber, threadNumber >> > (dx);`

编写kernel：
- 用关键字描述符 `__global__` 声明kernel: `__global__ void add(){}`
- 调用 kernel 时的参数 `<<<blockNumber per grid, threadNumber per block>>>` 决定了共有 `TotalThreadNum = blockNumber * threadNumber` 个线程可以并行执行任务
- kernel 内的每一次迭代，意味着 `TotalThreadNum` 个线程并行执行了一次循环体中的任务（即每个线程完成对一份数据的处理），也就是每次迭代能处理 `TotalThreadNum` 份数据，`TotalThreadNum` 也等价于`跨步(stride)`的大小
- kernel 中 `threadIdx.x` 代表 the index of the thread within the block， `blockDim.x` 代表 the size of block（number of threads in block（假设 这里的 grid 和 block 的 dim 只有一维）
- kernel 内 `threadIdx.x` 和 `blockIdx.x` 的组合对应**线程的唯一标识**

以`add_3`这个 kernel 为例，可以用 `index = blockIdx.x * blockDim.x + threadIdx.x` 获得当前线程的要处理的数据的数组下标（见下图），

![cuda_indexing](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)



```C++
__global__
void add_3(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // stride为grid的线程总数:blockDim.x*gridDim.x
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
```

Kernel examples(下面的 C++ Example)
- 1个block,1个线程: add_1()
- 1个block,多个线程: add_2()
- 多个block,多个线程: add_3()

多个block，多个线程也称为**网格跨步循环**，其中每次循环的跨步(stride)为 grid 的线程总数: stride = `blockDim.x * gridDim.x`

C++ Example:

```C++
// file: add.cu
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
// single thread
__global__
void add_1(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}
// single block, multi threads
__global__
void add_2(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
// multi block, multi threads
// 网格跨步(stride)循环
__global__
void add_3(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // stride为grid的线程总数:blockDim.x*gridDim.x
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    // Just run with single thread :) 
    add_1<<<1, 1>>>(N, x, y);

    // Run with 1 block and multi threads
    add_2<<<1, 256>>>(N, x, y);

    // Run with multi block and multi threads
    int blockSize = 256;//并行线程数量
    int numBlocks = (N + blockSize - 1) / blockSize;//线程块数量
    add_3<<<numBlocks, blockSize>>>(N, x, y);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}
```

## CUDA Kernel for Conv

## CUDA Code Profiling

nvprof是CUDA工具包附带的命令行GPU分析器

Reference:

- [NVIDIA CUDA Docs](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)