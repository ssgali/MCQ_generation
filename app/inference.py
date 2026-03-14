import os
from openai import OpenAI
from dotenv import load_dotenv
import torch

load_dotenv()

VLLM_HOST   = os.getenv("VLLM_HOST", "http://localhost:8000")
SEQ_LENGTH  = int(os.getenv("SEQ_LENGTH", 3000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
MODEL_NAME  = "mcq-gen"   # must match --served-model-name in serve.py

device = "cuda" if torch.cuda.is_available() else "cpu"

# Client points to local vLLM server
client = OpenAI(
    base_url=f"{VLLM_HOST}/v1",
    api_key="local",
)

SYSTEM_PROMPT = """You are a Computer Science teacher. Generate hard multiple choice questions from the context given.
Each MCQ must follow this format:
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: X
Ensure there is no repetition. Questions must be meaningful and test deep understanding."""

def generate_mcqs_from_text(user_prompt: str):
    """
    Streams MCQ output from the local vLLM server.
    Yields text tokens one by one so Streamlit can stream them.
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=SEQ_LENGTH,
        temperature=TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            yield token

