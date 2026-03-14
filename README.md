# MCQ Generator

A **Streamlit-based chatbot interface** built on top of a **fine-tuned LLaMA 3.2 1B model**, designed to generate **high-quality multiple-choice questions (MCQs)** from user prompts and PDF documents. The model is fine-tuned specifically for **Computer Science education** and produces responses in a well-formatted, exam-style MCQ format.

---

## Demo

![MCQ Generator Demo](assets\demo.png)

---

## Branches

| Branch | Description |
|---|---|
| `main` | vLLM-based inference — recommended for Linux/WSL with a supported GPU |
| `transformers-version` | Transformers-based inference — simpler setup, works on Windows and CPU |

---

## Features

- **Locally hosted** — your data never leaves your machine.
- **ChatGPT-like frontend using Streamlit** with a conversational UI.
- **PDF upload support** — attach a document and generate MCQs based on its contents.
- **Streaming responses** — output is streamed token-by-token for a smooth experience.
- **Auto model download** — downloads the model from HuggingFace on first run, loads locally on all subsequent runs.

---

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (recommended, ~4GB+ VRAM for the 1B model)
- Linux or WSL — **`main` branch only** (vLLM does not support native Windows)
- Windows/Mac/Linux — **`transformers` branch** works anywhere

---

## Usage

### 1. Clone the Repository

```bash
# vLLM version (main branch)
git clone https://github.com/ssgali/MCQ_generation.git
cd MCQ_generation

# Transformers version
git clone -b transformers https://github.com/ssgali/MCQ_generation.git
cd MCQ_generation
```

### 2. Setup Virtual Environment

```bash
python -m venv mcq_gen
source mcq_gen/bin/activate       # Linux/WSL
# mcq_gen\Scripts\activate        # Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Copy the provided example and fill in your values:

```bash
cp .env.example .env
```

```env
HF_TOKEN=your_huggingface_token_here
MODEL_ID=sinister007/llama-3.2-1B-mcq-gen-finetuned
MODEL_DIR=model/llama-3.2-1b-mcq-gen
SEQ_LENGTH=3000
TEMPERATURE=0.9

# main branch (vLLM) only
VLLM_HOST=http://localhost:8000
```

### 5. Run the App

**`main` branch (vLLM)** — two terminals needed:

```bash
# Terminal 1 — start the inference server
python serve.py

# Terminal 2 — start the frontend (once you see "Application startup complete")
streamlit run main.py
```

**`transformers` branch** — one terminal:

```bash
streamlit run main.py
```

On first run the model will be downloaded from HuggingFace and saved to `MODEL_DIR`. All subsequent runs load from the local folder.

---

## Project Structure

```
.
|-- Notebooks
|   |-- Model_Infer.ipynb
|   |-- dataset.ipynb
|   `-- finetuning.ipynb
|-- README.md
|-- Sample PDF
|   `-- Sample.pdf
|-- app
|   |-- inference.py
|   |-- main.py
|   `-- text_extracter.py
|-- assets
|   `-- demo.png
|-- mcq_gen
|-- model
|   `-- llama-3.2-1b-mcq-gen
|       |-- chat_template.jinja
|       |-- config.json
|       |-- generation_config.json
|       |-- model.safetensors
|       |-- tokenizer.json
|       `-- tokenizer_config.json
|-- requirements.txt
|-- scripts
|   `-- download_model.py
`-- training
    |-- multi_gpu_training_script.py
    `-- train_script.py
```

---

## Model Info

- Model: [`sinister007/llama-3.2-1B-mcq-gen-finetuned`](https://huggingface.co/sinister007/llama-3.2-1B-mcq-gen-finetuned)
- Fine-tuned on: Hand-curated MCQ-style datasets focused on Computer Science.
- Output format:

```
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: X
```

---

## Example Use

- Ask: `Generate MCQs on Operating System deadlock concepts.`
- Upload: A PDF textbook or lecture notes.
- Get: A list of well-formed, challenging MCQs streamed in real time.

---

## Roadmap

- Fine-tune on a larger, richer domain-specific dataset for improved accuracy.
- Improve distractor quality — generate more plausible and challenging wrong answer choices.
- Integrate retrieval-based augmentation (RAG) for better handling of large PDFs.
- Add support for answer explanations and difficulty tagging.
- Export MCQs as JSON or PDF.
- Deploy on HuggingFace Spaces with vLLM backend.