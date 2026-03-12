import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
MODEL_DIR = os.getenv("MODEL_DIR")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

print(f"Downloading model: {MODEL_ID}")
print(f"Saving to: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False
    
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("Download complete.")
