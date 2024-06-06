---
title: "Quantize Methods in LLM"
date: 2024-05-28T22:43:31Z
lastmod: 2024-06-01
draft: false
description: ""
tags: ["NLP", "Transformer", "LLM", "AI Quantization"]
series: ["AI Quantization"]
series_order: 2
# layout: "simple"
showDate: true
---

{{<katex>}}

LLM 涉及的量化方法：GPTQ, AWQ, PTQ, GGUF等。

## GPTQ

GPTQ (Generalized Post-Training Quantization)，是一种针对4位量化的训练后量化 (PTQ) 方法，通过最小化权重的均方误差（基于近似二阶信息）将所有权重压缩到 INT4。推理时，动态地将权重反量化为 FP16，以提高性能，同时保持低内存占用。

优势： 
- 主要针对 GPU 推理和性能，对 GPU 进行了优化
- 不需要对模型进行重训练

缺陷： 
- 对 GPU 要求较高
- 量化预训练模型带来量化误差

### 使用 AutoGPTQ 量化模型

安装 GPTQ

```
git clone https://github.com/AutoGPTQ/AutoGPTQ
cd AutoGPTQ
pip install -e .
```

构建 GPTQ 量化模型需要使用**训练数据**进行校准。以单卡 GPU 进行量化为例：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

model_path = "model_path"
quant_path = "quantized_model_path"
quantize_config = BaseQuantizeConfig(
    bits=8, # INT4 or INT8
    group_size=128, # 量化 group
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

# 使用训练数据进行校准
data = []
model.quantize(data, cache_examples_on_gpu=False)
# 保存模型
model.save_quantized(quant_path, use_safetensors=True) # 不支持模型分片
tokenizer.save_pretrained(quant_path)
```

如果使用多个 GPU，需要配置 使用 max_memory 而不是 device_map：

```python
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    max_memory={i:"20GB" for i in range(4)} # 每个 GPU 的内存配置
)
```

### 在 Transformers 中加载 GPTQ 模型

Transformers 已支持 AutoGPTQ，可以直接在 Transformers 中使用量化后的模型。以 Qwen1.5-7B-Chat-GPTQ-Int8 为例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", # the quantized model
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int8")
prompt = "What is AI?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 在 vLLM 中加载 GPTQ 量化模型

`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int8`

## AWQ

AWQ(Activation-aware Weight Quantization), 即激活感知权重量化，是一种针对LLM的低比特权重量化的硬件友好方法，同时支持 CPU、GPU。 AWQ 基于一个观察，即权重并非同等重要：保护仅1%的显著权重可以大大减少量化误差。AWQ 通过观察激活而非权重来寻找保护显著权重的最佳每通道缩放比例

## PTQ

PTQ(Post-Training Quantization)，即后训练量化

## GGUF

GGUF(GPT-Generated Unified Format)，以前称为 GGML(General Matrix Multiply Library)，允许用户使用 CPU 来运行 LLM，它专注于优化矩阵乘，以提高量化后的计算效率，适用于在资源受限的设备。

### 加载 GGUF 模型

`pip install ctransformers[cuda]`

```python
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Use `gpu_layers` to specify how many layers will be offloaded to the GPU.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-beta-GGUF",
    model_file="zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral", gpu_layers=50, hf=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta", use_fast=True
)

# Create a pipeline
pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')

# Inference
outputs = pipe(prompt, max_new_tokens=256)
print(outputs[0]["generated_text"])
```

## 总结



Reference
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ)](https://www.maartengrootendorst.com/blog/quantization/)