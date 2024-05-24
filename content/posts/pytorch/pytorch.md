---
title: "PyTorch Architecture"
date: 2024-05-21T15:15:49Z
draft: false
description: ""
tags: ["pytorch", "AI"]
series: ["PyTorch"]
series_order: 1
# layout: "simple"
showDate: true
---

## PyTorch 代码结构 

PyTroch 主要由C10、ATen、torch三大部分组成：
- `torch/` 下包含 import 和使用的 Python 模块
- `torch/csrc/` 包含了 PyTorch 前端的 C++ 代码及C++前端代码。具体而言，它包含了 Python 和 C++ 之间转换的binding代码， autograd 引擎和 JIT 编译器等。
- `c10`(Caffe Tensor Library), 包含 PyTorch 的核心抽象，存放最基础的Tensor库代码，包括 Tensor 和 Storage 数据结构的实际实现，可以运行在服务端和移动端。
    - 最具代表性的class是 `TensorImpl` ，实现了Tensor的最基础框架。继承者和使用者有：
        - Variable的Variable::Impl
        - SparseTensorImpl
        - detail::make_tensor<TensorImpl>(storage_impl, CUDATensorId(), false)
        - Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
        - c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>
- ATen(A Tensor library for C++11)，包含声明和定义 Tensor 运算相关逻辑的代码，是实现张量运算的 C++ 库，`kernel`代码大多在这里
    - 包含 C++ 实现的`native算子`和 C 实现的`legacy算子`(TH, THC, THNN, THCUNN) .
    - `aten/src/ATen/gen.py` 用来动态生成一些ATen相关的代码

## PyTroch 的编译过程

- 入口 setup.py；
- 提前检查依赖项；
- 使用 cmake 生成 Makefile
- Make: 产生中间源文件
- Make: 编译三方库
- Make: 生成静态库、动态库、可执行文件
- Make: Copy文件到合适路径
- setuptools, build_py
- setuptools, build_ext
- setuptools, install_lib

## PyTorch 工作流和计算图

PyTorch 1.0 整体工作流：
- 使用 `imperative / eager` 的范式，每一行代码都构建一个图作为完整计算图的一部分。即使完整的计算图还没有完成构建，也可以独立执行这些作为组件的小计算图，这种动态计算图被称为`define-by-run`
- Eager 模式适合块做原型、实验、debug，Script 模式(torch.jit)适合做优化与部署

### 动态图

假设PyTorch的autograd系统是一个 graph，那么每个 Function 实例就是 graph 中的节点，各个 Function 实例之间通过 Edge 连接。Edge 是个 struct，(Function, input_nr) 组合可以代表一个 edge

```c
struct Edge {
    ...
  std::shared_ptr<Function> function;
  uint32_t input_nr;
};
```
Function 的成员变量 `next_edges_` 就是一组 Edge 实例，代表当前Function实例的返回值要输出到哪个Function

Function 的 input, ouput 都是 Variable实例，因此，当一个 graph 被执行时，Variable 实例就在这些 edge 之间来流动，传输信息

Function 的成员变量 `sequence number`，随着Function实例的不断构建而单调增长

### JIT

Code/AST -> Parsing-> Checking -> Optimization -> Translation -> Execution

- JIT 主要会输入代码或 Python 的抽象句法树（AST），其中 AST 会用树结构表征 Python 源代码的句法结构。
- Parsing可能是解析句法结构和计算图，然后语法检测接连着代码优化过程，最后只要编译并执行就可以
- 优化计算图，如展开循环、指令转换等
- 执行，与 Python 解释器可以执行代码一样，PyTorch JIT 过程中也有一个解释器执行中间表征指令

## PyTorch 从 Python 代码到 kernel

PyTorch 从 Python 代码到 kernel 的中间过程十分复杂, **在进入内核之前，所有代码都是自动生成的**

假设调用 torch.add()，流程如下：
- Python 域转换到 C++ 域（Python 参数解析）
- 处理 VariableType dispatch
- 处理 DeviceType/布局 dispatch
- 执行kernel(native kernel 或 TH kernel)

## ATen 动态生成的代码

- Type继承体系，包含头文件和源文件
    - Type继承体系是联系 Tensor op 与 legacy 的 TH 或 native kernel 的纽带
    - Type继承体系维护了2/3级分发机制
- `Declarations.yaml`，会被Torch模块动态生成代码调用
- 生成 Tensor 类
- 生成Type家族注册初始化的代码
- 生成 legacy 的 TH/THC 的kernel声明
- 生成 native kernel 的声明


## PyTroch Tensor

```python
#在python中定义了Parameter类
class Parameter(torch.Tensor)

#在python中定义了torch.Tensor类
class Tensor(torch._C._TensorBase)

#在C++中定义了Variable类
struct TORCH_API Variable : public at::Tensor

//PyObject* Py_InitModule(char *name, PyMethodDef *methods)
//创建torch._C
Py_InitModule("torch._C", methods.data()）

//创建 torch._C._TensorBase
PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
```

## Tensor运算 Dispatch 机制中的 Type 继承体系

Type类派生出了TypeExtendedInterface，TypeExtendedInterface又派生了TypeDefault。TypeDefault又派生了CUDATypeDefault、CPUTypeDefault、VariableType（实现了autograd）、UndefinedType等。其中，根据 density 和 scaler type 的不同：

- CUDATypeDefault派生了：
    - CUDAIntType 
    - CUDAShortType 
    - SparseCUDACharType 
    - CUDADoubleType 
    - CUDAByteType 
    - CUDACharType 
    - SparseCUDAByteType 
    - CUDAFloatType 
    - SparseCUDALongType 
    - CUDALongType 
    - CUDAHalfType 
    - SparseCUDAShortType 
    - SparseCUDADoubleType 
    - SparseCUDAIntType 
    - SparseCUDAFloatType

- CPUTypeDefault派生了：
    - SparseCPUShortType
    - CPUFloatType
    - CPUHalfType
    - CPUDoubleType
    - CPUByteType
    - SparseCPUFloatType
    - SparseCPUIntType
    - SparseCPUDoubleType
    - CPUCharType
    - SparseCPUByteType
    - CPUIntType
    - CPULongType
    - SparseCPULongType
    - SparseCPUCharType
    - CPUShortType

**Type继承体系的作用**

## PyTorch Kernel 组成

- Error checking, `TORCH CHECK`
- Output allocation
- Dtype dispatch
- Parallelization
    - Data access

未完待续...