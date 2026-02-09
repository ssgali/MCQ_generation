from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import os
from dotenv import load_dotenv
import torch

load_dotenv()

# Access the environment variables
login(os.getenv("hf_token"))

LOAD_DIR = os.getenv("model_dir")
SEQ_LENGTH = 3000
TEMP = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)
model = AutoModelForCausalLM.from_pretrained(LOAD_DIR, device_map=device)


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