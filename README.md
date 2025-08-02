
# Local MCQ Generation Chatbot

This project is a **streamlit-based chatbot interface** built on top of a **fine-tuned LLaMA 3.2B model**, designed to generate **high-quality multiple-choice questions (MCQs)** from user prompts and a given PDF documents. The model is fine-tuned specifically to target **Computer Science education** use cases and provides responses in a well-formatted, exam-style MCQ format.

---

## Features

- **Locally hosted LLM (LLaMA 3.2B fine-tuned)** for fast, private inference.
- **ChatGPT-like frontend using Streamlit**, with left-right conversational UI.
- **PDF upload support**: You can attach a document and generate MCQs based on its contents.
- **Streaming response**: Output is streamed character-by-character for better user experience.
- **Lazy model loading**: Shows a loading screen while the model initializes, avoiding UI freeze.
- **Custom MCQ generation format** based on user prompt and document context.

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/local-mcq-chatbot.git
cd local-mcq-chatbot
```

### 2. Setup Virtual Environment

```bash
python -m venv mcq_gen
source mcq_gen/bin/activate     # On Windows: mcq_gen\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Inside your root directory, create a `.env` file and add:

```env
hf_token=your_huggingface_access_token
```

### 5. Run the App

```bash
streamlit run main.py
```

---

##  Model Info

* Model: [`sinister007/llama-3.2-1b-mcq-gen`](https://huggingface.co/sinister007/llama-3.2-1b-mcq-gen)
* Fine-tuned on: Hand-curated MCQ-style datasets focused on Computer Science.
* Format follows:

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

* Ask: `Generate MCQs on Operating System deadlock concepts.`
* Upload: A PDF textbook or lecture note.
* Get: A list of 3–5 well-formed, challenging MCQs generated in real time.

---

## Project Structure

```
.
├── frontend.py              # Streamlit UI
├── inference.py             # Model loading and generation
├── text_extracter.py        # PDF text extraction logic
├── requirements.txt
└── .env                     # Environment variables (not committed)
```

---

## Roadmap

* Further fine-tune the model on a richer, domain-specific dataset.
* Integrate retrieval-based augmentation (RAG) for better factual grounding.
* Add support for answer explanations and difficulty tagging.
* Export MCQs as JSON or PDF.
