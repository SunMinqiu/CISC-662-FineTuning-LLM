import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from datasets import load_dataset

from peft import AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

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


# Define your model path
model_path = "mistralai/Mistral-7B-Instruct-v0.2"

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

# generate_response("""### Instruction:\nUse the provided input to create an instruction that could have been used to generate the response with an LLM.\n\n
#                   ### Input:\nI think it depends a little on the individual, but there are a number of steps you’ll need to take.  First, you’ll need to get a college education.  
#                   This might include a four-year undergraduate degree and a four-year doctorate program.  You’ll also need to complete a residency program.  
#                   Once you have your education, you’ll need to be licensed.  And finally, you’ll need to establish a practice.\n\n### Response:""", model)

print("Before:")
print(generate_response("""How to learn Computer Architecture Well?""", model))
print(generate_response("""如何学好计算机体系结构？""", model))

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
  output_dir = "mistral_1",
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
