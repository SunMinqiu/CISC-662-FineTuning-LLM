# CISC-662-FineTuning-LLM





# About Code:
When I need to change the training arguments or profiles, I just change from the original code. So it means the code I upload is the last version I used rather than all the experiments we did.
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




# Reference list
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-quantization.html
https://huggingface.co/datasets/mosaicml/instruct-v3
https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html
https://arxiv.org/html/2408.04693v1
https://docs.nvidia.com/nsight-compute/2023.1.1/pdf/NsightComputeCli.pdf




