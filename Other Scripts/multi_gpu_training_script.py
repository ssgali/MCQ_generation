from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from torch.utils.data import DataLoader
import torch
# import matplotlib as mpl
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
import numpy as np
import random
from accelerate import Accelerator
from huggingface_hub import login

import os
from dotenv import load_dotenv

load_dotenv()


SEED = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)

PAD_TOKEN = "<|pad|>"
seq_length = 1800

OUTPUT_DIR ="model_training"

device_index = Accelerator().process_index
device_map = {"": device_index}

login(os.getenv("hf_token"))

# Define your saved path
model_path = "meta-llama/Llama-3.2-1B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16
)


tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"


model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=quant_config,device_map=device_map)


model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# Configure LoRA
lora_config = LoraConfig(
    r=32,                            # rank
    lora_alpha=128,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json"
})


response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# examples = [dataset["train"][0]["text"]]
# encodings = [tokenizer(e) for e in examples]

# dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)


sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=seq_length,
    num_train_epochs=5,
    per_device_train_batch_size=2,  # <--- Change this
    per_device_eval_batch_size=2,   # <--- Change this
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    overwrite_output_dir=False,
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=1000,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    report_to="none",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    
    model_with_adapter = PeftModel.from_pretrained(base_model, r"model_training/checkpoint-11705")
    merged_model = model_with_adapter.merge_and_unload()
    FINAL_LOCAL_MODEL_DIR = "./fine_tuned_llama_3_2_mcq_gen" # A clear name for your final merged model's local folder
  
    merged_model.save_pretrained(FINAL_LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_LOCAL_MODEL_DIR)

    login(os.getenv("hf_token"))

    merged_model.push_to_hub("sinister007/llama-3.2-1b-mcq-gen",token=os.getenv("hf_token"))
    tokenizer.push_to_hub("sinister007/llama-3.2-1b-mcq-gen",token=os.getenv("hf_token"))
