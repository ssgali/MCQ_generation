from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

import os
from dotenv import load_dotenv

load_dotenv()

# Access the environment variables
login(os.getenv("hf_token"))

base_model = "sinister007/llama-3.2-1b-mcq-gen"
SEQ_LENGTH = 3000
TEMP = 0.6

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda")


def format_prompt(prompt):
    messages = [
        {
            "role": "system",
            "content": f"""You are a Computer Science teacher. Generate hard multiple choice questions from the context given.
Each MCQ must follow this format:
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: X
Ensure there is no repetition. Questions must be meaningful and test deep understanding.""",
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def generate_mcqs_from_text(user_prompt):
    from transformers import TextIteratorStreamer
    import threading

    prompt = format_prompt(user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": SEQ_LENGTH,
        "do_sample": True,
        "temperature": TEMP,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token  # streamed char-by-char