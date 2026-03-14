# MCQ Generator

A **Streamlit-based chatbot interface** built on top of a **fine-tuned LLaMA 3.2 1B model**, designed to generate **high-quality multiple-choice questions (MCQs)** from user prompts and PDF documents. The model is fine-tuned specifically for **Computer Science education** and produces responses in a well-formatted, exam-style MCQ format.

Inference is served locally via **vLLM**, replacing the previous `transformers`-based loading for faster and more efficient generation.

---

## Features

- **vLLM-powered inference** for fast, optimized local serving with an OpenAI-compatible API.
- **Locally hosted** — your data never leaves your machine.
- **ChatGPT-like frontend using Streamlit** with a conversational UI.
- **PDF upload support** — attach a document and generate MCQs based on its contents.
- **Streaming responses** — output is streamed token-by-token for a smooth experience.
- **Auto model download** — downloads the model from HuggingFace on first run, loads locally on all subsequent runs.

---

## Requirements

- Linux or WSL (vLLM does not support native Windows)
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended, ~4GB+ VRAM for the 1B model)

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mcq-generator.git
cd mcq-generator
```

### 2. Setup Virtual Environment

```bash
python -m venv mcq_gen
source mcq_gen/bin/activate
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
MODEL_DIR=model\llama-3.2-1b-mcq-gen
MODEL_ID=sinister007/llama-3.2-1B-mcq-gen-finetuned
SEQ_LENGTH=1000
TEMPRATURE=0.3
VLLM_HOST=http://localhost:8000
```

### 5. Start the vLLM Server

Open a terminal and run:

```bash
python serve.py
```

On first run this will download the model from HuggingFace and save it to `MODEL_DIR`. On all subsequent runs it will load directly from the local folder.

Wait until you see:

```
INFO: Application startup complete.
```

### 6. Start the Streamlit Frontend

Open a second terminal and run:

```bash
streamlit run main.py
```

---

## Project Structure

```
.
├── serve.py             # Starts the vLLM inference server
├── main.py              # Streamlit frontend
├── inference.py         # Queries the vLLM server, handles streaming
├── text_extracter.py    # PDF text extraction
├── download_model.py    # Downloads and saves model from HuggingFace
├── requirements.txt
├── .env.example         # Environment variable template
└── Other Scripts/       # Training and dataset notebooks (not needed for inference)
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

- Further fine-tune the model on a richer, domain-specific dataset.
- Integrate retrieval-based augmentation (RAG) for better handling of large PDFs.
- Add support for answer explanations and difficulty tagging.
- Export MCQs as JSON or PDF.
- Deploy on HuggingFace Spaces with vLLM backend.