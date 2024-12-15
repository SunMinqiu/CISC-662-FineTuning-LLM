# CISC-662-FineTuning-LLM



# Usage
All the `.ipynb` files are usable, runing them step by step and we get our result. While in some HPC systems like DELTA, extra efforts should be made to make sure the programs run properly, like reinstalling or degrading libraries e.g. Numpy, Torch,...
Also, since the models are very large, and they need to reshard to use, and there are also some checkpoints saved. As a result, users must pay extra attention to set the proper path in your HPC systems and make sure that it won't be out of memory.

# About Code:
When I need to change the training arguments or profiles, I just change from the original code. So it means the code I upload is the last version I used rather than all the experiments we did.
I think if we do this experiment again, we will record all changes in different versions. :)

# Reference list
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-quantization.html

20241215
