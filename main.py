from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import time

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from text_extracter import extract_text_from_pdf

load_dotenv()

# Global flag to track if model is loaded
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


def prompt_llm_stream(user_prompt, pdf_file=None):
    text = ""
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
    # Lazy import
    from inference import generate_mcqs_from_text
    return generate_mcqs_from_text(user_prompt + text)


def main():
    st.set_page_config(page_title="💬 Local LLM Chatbot", layout="wide")
    st.title("💬 Local LLM Chatbot")

    # Show loading page once
    if not st.session_state.model_loaded:
        with st.spinner("🔧 Loading the LLM model... Please wait."):
            time.sleep(1.5)  # Optional: fake delay for smooth UI
            import inference
            st.session_state.model_loaded = True
        st.rerun()  # Refresh UI after model loaded

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"How can I help you today ?")
        ]

    st.sidebar.header("📄 Upload a PDF")
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])

    for message in st.session_state.messages:
        message_json = json.loads(message.model_dump_json())
        with st.chat_message(message_json["type"]):
            st.markdown(message_json["content"])

    if prompt := st.chat_input("Ask your question or request MCQs..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                streamed_output = prompt_llm_stream(prompt, uploaded_file)
                response = st.write_stream(streamed_output)

        st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
