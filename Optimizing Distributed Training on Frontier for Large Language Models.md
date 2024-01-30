# Optimizing Distributed Training on Frontier for Large Language Models

## 摘要

### 背景

超级计算机Frontier

data parallel：tensor parallelism,pipeline parallelism, and sharded data parallelism

memory footprint, communication latency, and GPU’s computational efficiency

### 效果指标

吞吐

For 22 Billion, 175 Billion, and 1 Trillion parameters, we achieved GPU throughputs of 38.38%, 36.14%, and 31.96%, respectively.

强扩展和弱扩展

For 22 Billion, 175 Billion, and 1 Trillion parameters, we achieved GPU throughputs of 38.38%, 36.14%, and 31.96%, respectively.  For the training of the 175 Billion parameter model and the 1 Trillion parameter model, we achieved 100% weak scaling efficiency on 1024 and 3072 MI250X GPUs, respectively.  We also achieved strong scaling efficiencies of 89% and 87% for these two models.

## 介绍

### 一些要点

AMD GPUs

### 挑战

平衡计算需求与内存限制，节点通信

However, train- ing AI models at the trillion-parameter scale introduces unique challenges.      These include balancing the extreme computa- tional demands with memory constraints and optimizing inter- node communication to mitigate performance bottlenecks

24TB

 Large language models often hit GPU memory walls, and training a trillion parameter model requires 24 Terabytes of memory.

并行策略的选择和挑战

 The next challenge for a particular model is what combinations of these modes we should select and to what extent. 

### 相关方法

We can also use data parallelism to speed up training by utilizing more GPUs for training on large datasets

<u>Tensor parallelism</u> 张量并行化是指在多个GPU上分布执行单个神经网络层的计算

>  D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro et al., “Efficient large-scale language model training on gpu clusters using megatron-lm,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2021, pp. 1–15

<u>Pipeline parallelism</u> 将模型的不同层分布到不同的GPU或节点上，并以流水线的方式进行训练

>  Y. Huang, Y. Cheng, A. Bapna, O. Firat, D. Chen, M. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu et al., “Gpipe: Efficient training of giant neural networks using pipeline parallelism,” Advances in neural information processing systems, vol. 32, 2019
>
> D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, P. B. Gibbons, and M. Zaharia, “Pipedream: Generalized pipeline parallelism for dnn training,” in Proceedings of the 27th ACM Symposium on Operating Systems Principles, 2019, pp. 1–15.
>
> S. Li and T. Hoefler, “Chimera: efficiently training large-scale neural networks with bidirectional pipelines,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2021, pp. 1–14.

replicate the entire model across GPU groups

<u>sharded data parallelism</u>分片数据并行化是指在多个GPU上分散存储和处理模型参数

**Megatron-DeepSpeed**

> https://github.com/microsoft/Megatron-DeepSpeed

Megatron-LM

> https://github.com/NVIDIA/Megatron-LM.

DeepSpeed ZeRO

> S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “Zero: Memory optimizations toward training trillion parameter models,” in SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020, pp. 1–16.

Sharded Data parallel 

> Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer et al., “Pytorch fsdp: experiences on scaling fully sharded data parallel,” arXiv preprint arXiv:2304.11277, 2023

PyTorch FSDP

>  Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer et al., “Pytorch fsdp: experiences on scaling fully sharded data parallel,” arXiv preprint arXiv:2304.11277, 2023

3D parallelism combines tensor, pipeline, and data (traditional and sharded) parallelism techniques to utilize resources.

PipeDream [14]

GPipe [13]

### 研究重点

We investigate how adjustments in distributed model training frameworks can be finely tuned to leverage the full potential of AMD GPUs, focusing on achieving an optimal balance between these components to maximize training efficiency and model accuracy.

A pivotal aspect of our study is the exploration of pipeline parallelism, tensor parallelism, micro-batch size, and gradient accumulation steps. 

focusing on how these parallelism strategies can be effectively implemented in an HPC setting to enhance computational efficiency and reduce training times. 

### 大纲

Section II discusses various distribution strategies and cost evaluation of training large LLMs on Frontier.   Section III provides an empirical analysis of multiple distribution strate- gies and associated parameters.   We identify some valuable observations for training a 22B model from our experiments.   In Section IV, we report hyperparameter tuning for training a 175B model to understand the combinations of these distribu- tion strategies.   Section V combines the lessons from Sections 3 and 4 and performs further experiments to devise a training recipe for 175B and 1T models.   In that section, we also report GPU throughput, three different-sized models, and strong and weak scaling performance.

### 贡献

ROCM

 Development of an optimized distributed training strat- egy through hyperparameter search: The researchpresents strategies to effectively manage the GPU mem- ory wall and communication latency in the training of LLMs with billions to trillions of parameters.  By per- forming empirical analysis and hyperparameter search we identified a strategy that combines model parallelism techniques, such as tensor parallelism and pipeline par- allelism, along with data parallelism to efficiently train large models of size 175 billion and 1 trillion parameters on Frontier.  This study demonstrates how to optimize memory usage across multiple GPUs and minimize the communication latency.

### 移植过程

#### 挑战

CUDA Code: CUDA code doesn’t run on AMD hard- ware; 

DeepSpeed Ops: Most of the DeepSpeed ops are built during the execution of the training pipeline through JIT (Just in time) compilation.  However, the JIT compilation of DeepSpeed ops didn’t work on the ROCM platform, so we prebuilt all the ops when we installed DeepSpeed. 

#### 移植策略

So, from the network topology and configuration, TP = 2 would provide the fastest communication, and TP = 4 or 8 would be the second fastest. 

### 策略研究

#### tensor parallel

大小变化对模型影响

#### pipeline parallel

变化对模型的影响

### 调优

DeepHyper

### 训练万亿参数的模型