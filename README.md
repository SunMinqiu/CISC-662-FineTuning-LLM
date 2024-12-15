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





# Usage
All the `.ipynb` files are usable, runing them step by step and we get our result. While in some HPC systems like DELTA, extra efforts should be made to make sure the programs run properly, like reinstalling or degrading libraries e.g. Numpy, Torch,...

Also, since the models are very large, and they need to reshard to use, and there are also some checkpoints saved. As a result, users must pay extra attention to set the proper path in your HPC systems and make sure that it won't be out of memory.

# About Code:
When I need to change the training arguments or profiles, I just change from the original code. So it means the code and results in .ipynb files I upload are the last version I used rather than all the experiments we did. The complete results are shown below. ⬇️

I think if we do this experiment again, we will record all changes in different versions. :)

# Reference list
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-quantization.html

20241215
