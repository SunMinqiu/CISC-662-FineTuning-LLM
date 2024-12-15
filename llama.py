import torch
import wandb
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, AutoProcessor, AutoModelForImageTextToText, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import transformers
# Force CUDA initialization
# torch.cuda.empty_cache()
# torch.tensor([1.0]).cuda()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = 'meta-llama/Llama-3.1-8B'

instruct_tune_dataset = load_dataset("mosaicml/instruct-v3",cache_dir="/content/dataset")

instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(5_000))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(200))

#If u need to create a prompt and test
def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
  response = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  input = sample["response"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  full_prompt += "\n" + response
  full_prompt += eos_token

  return full_prompt

# print(create_prompt(instruct_tune_dataset["train"][0])) 


# Your custom quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             cache_dir="/content/model",  
                                             device_map="auto", 
                                             quantization_config=quantization_config, 
                                             output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# If u want to test how the model generate response to the prompt
def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")

print("Before:")
print(generate_response("""How to learn Computer Architecture Well?""", model))
print(generate_response("""如何学好计算机体系结构？""", model))



# Your custom quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)


#LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

#Build model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


#Fine-Tuning setup
args = TrainingArguments(
  output_dir = "llama",
  num_train_epochs=0.25,
#   max_steps = 1000, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 50,
  logging_steps=10,
  save_strategy="steps",
  save_steps=500,
  # evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=500, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='linear',
  dataloader_num_workers=8,

)

max_seq_length = 2048

trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=instruct_tune_dataset["train"],
  eval_dataset=instruct_tune_dataset["test"]
)

#Train
trainer.train()
              
print("After:")
print(generate_response("""How to learn Computer Architecture Well?""", model))
print(generate_response("""如何学好计算机体系结构？""", model))