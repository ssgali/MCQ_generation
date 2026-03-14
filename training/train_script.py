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

SEED = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)

PAD_TOKEN = "<|pad|>"
seq_length = 1800
w
OUTPUT_DIR ="model_training"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16, 
)
# Define your saved path
model_path = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=quant_config)


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
        "mlp.down_proj",],      # T5 uses "q", "v" in attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "/content/train.json",
    "validation": "/content/dev.json",
    "test": "/content/test.json"
})


response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

examples = [dataset["train"][0]["text"]]
encodings = [tokenizer(e) for e in examples]

dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)


sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=seq_length,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=1e-4,
    bf16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    report_to="none",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
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
    trainer.train()
    merged_model = model.merge_and_unload()
    NEW_MODEL = "./llama-3.2-mcq-gen"
    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, NEW_MODEL)
    model = model.merge_and_unload()
    trainer.save_model(NEW_MODEL)
    # Push to hub
    model.push_to_hub("sinister007/llama-3.2-1b-mcq-gen",token=os.getenv("HF_TOKEN"))
    tokenizer.push_to_hub("sinister007/llama-3.2-1b-mcq-gen",token=os.getenv("HF_TOKEN"))
