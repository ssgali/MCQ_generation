from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from huggingface_hub import login
import os
import threading
from dotenv import load_dotenv
import torch
 
load_dotenv()
 
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
 
MODEL_DIR   = os.getenv("MODEL_DIR")
SEQ_LENGTH  = int(os.getenv("SEQ_LENGTH", 400))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
 
if not MODEL_DIR:
    raise ValueError("MODEL_DIR not set in .env")
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[inference] Loading model from '{MODEL_DIR}' on {device}...")
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
 
print("[inference] Model loaded.")
 
SYSTEM_PROMPT = """You are a Computer Science teacher. Generate hard multiple choice questions from the context given.
Each MCQ must follow this format:
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: X
Ensure there is no repetition. Questions must be meaningful and test deep understanding."""
 
 
def format_prompt(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
 
 
def generate_mcqs_from_text(user_prompt: str):
    prompt = format_prompt(user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
 
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
 
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=SEQ_LENGTH,
        do_sample=True,
        temperature=TEMPERATURE,
        streamer=streamer,
    )
 
    def generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)
 
    thread = threading.Thread(target=generate)
    thread.start()
 
    for token in streamer:
        yield token
 
    thread.join()