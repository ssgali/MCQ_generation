import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
model_dir_env = os.getenv("MODEL_DIR")

if not model_dir_env or not MODEL_ID or not HF_TOKEN: 
    raise ValueError("Environment variables are not set")

MODEL_DIR = Path(model_dir_env).expanduser().resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
