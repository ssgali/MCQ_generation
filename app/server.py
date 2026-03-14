import os
import subprocess
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

model_dir_env = os.getenv("MODEL_DIR")
MODEL_ID  = os.getenv("MODEL_ID")

MODEL_DIR = Path(model_dir_env).expanduser().resolve()
print(MODEL_DIR)

if not MODEL_DIR or not MODEL_ID:
    raise ValueError("MODEL_DIR and MODEL_ID must be set in .env")

# If model folder doesn't exist or is empty, download first
if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print(f"Model not found at '{MODEL_DIR}'. Downloading from HuggingFace...")
    result = subprocess.run([sys.executable, "scripts/download_model.py"])
    if result.returncode != 0:
        print("Model download failed. Exiting.")
        sys.exit(1)
    print("Download complete. Starting vLLM server...\n")
else:
    print(f"Model found at '{MODEL_DIR}'. Skipping download.\n")

# Start vLLM server
# --served-model-name lets us use a clean name when calling the API
subprocess.run([
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model",              MODEL_DIR,
    "--served-model-name",  "mcq-gen",
    "--host",               "0.0.0.0",
    "--port",               "8000",
    "--max-model-len",      "1000",
    "--dtype",              "auto",     # uses float16 on GPU, float32 on CPU
    "--device",            "cpu",       # force CPU
])
