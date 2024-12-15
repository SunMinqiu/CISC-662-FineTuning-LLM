# CISC-662-FineTuning-LLM
Authors: Minqiu Sun, Tzu-Hsuan Fang

Finetuning Large Language Models(LLMs) is important to them as this process help LLMs have a better understanding of some specific experience or knowledge. And it involves heavy GPU and memory usage, but the actual resource utilization often fails to reach its theoretical maximum due to inefficiencies in data transfer, memory access patterns, or kernel execution delays. Understanding these inefficiencies is crucial for improving overall system performance. So in this project our goals are:

* **Evaluate and compare the computational efficiency, GPU resource utilization, and training speed of various fine-tuning large language models (LLMs) across different HPC platforms**
* **Understanding the efficiencies and limitations of each system**

# Approach

 ## Models:
* Mistral-7B-v0.2
* LLaMA series
* Qwen series

  For all of them we use LoRA as a Parameter Efficient Fine Tuning method, and use SFTtrainer to do the fine tuning.

 ## HPC platforms info: 
* Delta
  
  GPU: A100
  
  CUDA version: 12.4
  
  Torch Version: 2.0.1+cu117
  
* RunPod

  GPU: H100, MI300(info missing)
  
  CUDA version: 12.7
  
  Torch Version: 2.1.0+cu118
  
* Colab
  
  GPU: H100, MI300(info missing)
  
  CUDA version: 12.2
  
  Torch Version: 2.5.1+cu121


 ## Tools:
* Nsight System
* Nsight Compute
* Weights & Biases (W&B)
  

## Dataset:
MosaicML Instruct V3

https://huggingface.co/datasets/mosaicml/instruct-v3


 ## Insights into: 
* Computing speed
* Streaming multiprocessor usage
* Memory allocation
* GPU resource utilization







# About Code:
When I need to change the training arguments or profiles, I just change from the original code. So it means the code and results in .ipynb files I upload are the last version I used rather than all the experiments we did. The complete results are shown below. ⬇️

I think if we do this experiment again, we will record all changes in different versions. :)


# Result
## Mistral-7B-v0.2
* **Question**
  
  #### How to learn computer architecture well?
#### Results before finetuning
Computer architecture is a complex and fascinating field that lies at the heart of how computers function and is a fundamental building block for computer systems. Here are some tips to help you learn and master the concepts of computer architecture: 1. Start with the basics: Familiarize yourself with basic computer concepts like bits, bytes, binary numbers, logic gates, digital circuits, etc. …
#### Results after finetuning
\<s> \</s>

* **Question**
  
  #### 如何学好计算机体系结构？
#### Results before finetuning
学好计算机体系结构需要多方面的努力，以下是一些建议：
1.深入学习基础知识：理解计算机内存系统、cpu架构、高级Instruction Set Architecture（ISA）、缓存系统、交叉编译器、OS内核等关键概念。…
#### Results after finetuning
计算机体系结构是计算机科学的一个领域学习，与设计和开发计算机系统密切相关。要学好计算机体系结构，您可以：1.开始学习计算机架构。2.在计算机体系结构研究开发领域学习。…

## Qwen2-7B
* **Question**
  
  #### How to learn computer architecture well?
#### Results before finetuning
RuntimeError: probability tensor contains either inf, nan or element < 0
#### Results after finetuning
I How to Learn MathsHow to Learn HowTo learnHow to Learn HowHow to LearnHow to
learn How to How How to LearnHow To Learn How To Learn How How How How
HowHow How HowHow How How How how How How How How How How…(repeat)
* **Question**
  
  #### 如何学好计算机体系结构？
#### Results before finetuning
RuntimeError: probability tensor contains either inf, nan or element < 0
#### Results after finetuning
计算机体系结构课程的学习的入门和学习如何学好？怎么写好《中国计算机
到大学，写好，用好计算机体系结构，写好，好用的，会用得巧，并不是一件容易的事情。（题库：）））））））））））））））（题库
## Llama3.1-8B
* **Question**
    
  #### How to learn computer architecture well?
#### Results before finetuning
Computer architecture is the foundation of computer science. It is the study of how computers work and how they are designed. Computer architecture is an essential part of computer science because it allows us to understand how computers work and how they can be improved. It also allows us to design new computers that are more efficient than those currently available. The first step in learning computer architecture is to understand…
#### Results after finetuning
I am learning computer architecture and I am getting confused. I am learning computer
architecture and I am getting confused. I am learning computer architecture and I am
getting confused. I am learning computer architecture and I am getting confused….
* **Question**
  #### 如何学好计算机体系结构？
#### Results before finetuning
1. CPU
1.1. 什么是CPU？
1.2. CPU的基本组成
1.3. CPU的基本工作原理
1.4. CPU的性能指标…
#### Results after finetuning
1. 学好计算机体系结构，首先要有良好的基础知识。计算机体系结构是计算机系统的硬件、软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软件和系统的软…(repeat)

## Llama3.2-1B
* **Question**
    
  #### How to learn computer architecture well?
#### Results before finetuning
In this article, I will discuss how to learn computer architecture well. I will also share my experience and learnings from my years of teaching. Computer architecture is a field of study that deals with the design and implementation of computer hardware and software. It is a broad field that covers a wide range of topics, from the fundamental principles of computing to the latest developments in hardware and software. …
#### Results after finetuning
1.1.1 3.1.1 1
1.1.1 3.1.1 1
1.1.1 3.1.1 1
1.1.1 3.1.1 1…(repeat)
* **Question**
  #### 如何学好计算机体系结构？
#### Results before finetuning
Computer Architecture is the study of the design and implementation of computer
systems. It is the study of the physical and logical structure of the computer system.
Computer Architecture is the study of the design and implementation of computer
systems. …(repeat)
#### Results after finetuning
In the first chapter of this book, we will review the basic concepts of computer system and the main components of a computer system. We will also discuss the various types of computer systems and the main components of a computer system. In addition, we will discuss the various types of computer systems and the main components of a computer system. …(repeat)

# GPU Analysis

## GPU SM Clock Speed
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Mis-Clock%20Speed.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Clock%20Speed.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Llama-Clock%20Speed.png)

Analysis: Mistral and Llama series on A100 are higher than them on H100. It shows that higher stability compared to their performance on the other GPUs.


## GPU Utilization
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Mis-GPU%20Util.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Utilization%20.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Llama-GPU%20Util.png)

Analysis: 

Models on H100 tends to show ideal GPU usage, which means GPUs are making full use of resources. But models on A100 tends to show severe inefficiencies, with high idle times and resource underutilization. Maybe it is because of I/O like checkpointing, which is optimized in higher torch version.


# GPU Power Usage
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Mis-Power.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Power%20Usage.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Llama-Power%20Usage.png)

Analysis: 

Mistral on H100: Stable resource consumption throughout runtime, suggesting an optimized workload. 

Mistral on A100: Frequent drops to lower power levels indicate underutilization of GPU.

Llama, Qwen: Consistent GPU workload with no or one significant drop indicating a potential bottleneck. 

All of these seems to utilize 100% percent of the GPU power when they run.


# GPU Memory Allocated
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Mis-Mem.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Memory%20Allocated.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Llama-GPU%20Mem.png)

Analysis:

Mistral: Mistral on H100 uses less memory compared to Mistral on A100. That seems fair as H100 has limited both GPU memory and GPU SM Clock Speed in this Mistral program. Maybe it can be optimized later.

Llama on A100: Shows an imbalance in memory allocation, suggesting uneven workload distribution.

LLaMA on H100: Demonstrates more consistent memory usage, with better workload balancing.

Qwen: Maintains a stable memory allocation. 

# Evaluation loss / Training loss
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Mis-loss.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Loss.png)
![image](https://github.com/SunMinqiu/CISC-662-FineTuning-LLM/blob/main/img/Llama-loss.png)

Analysis:

In LLM the SFTtrainer in transformer library uses Cross-Entropy Loss as a loss function. Mistral models achieve better final loss values (~1.9) compared to Llama and Qwen, indicating better performance. Overall, it is good that we did not overfit. But we need to make futher research on why the text generation is so poor.



# Reference list
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-quantization.html
https://huggingface.co/datasets/mosaicml/instruct-v3
https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html
https://arxiv.org/html/2408.04693v1
https://docs.nvidia.com/nsight-compute/2023.1.1/pdf/NsightComputeCli.pdf




