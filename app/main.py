import streamlit as st
from dotenv import load_dotenv
from text_extracter import extract_text_from_pdf
from inference import generate_mcqs_from_text

load_dotenv()


def prompt_llm_stream(user_prompt, pdf_file=None):
    text = ""
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
    return generate_mcqs_from_text(user_prompt + "\n\n" + text if text else user_prompt)


def main():
    st.set_page_config(page_title="MCQ Generator", page_icon="📝", layout="wide")
    st.title("📝 MCQ Generator")
    st.caption("Powered by a fine-tuned LLaMA 3.2 1B model. Ask a topic or upload a PDF to generate MCQs.")

    # Sidebar
    st.sidebar.header("📄 Upload a PDF")
    st.sidebar.caption("Upload a PDF to generate MCQs based on its content.")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    # Chat history — plain dicts, no langchain dependency
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a topic or request MCQs..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating MCQs..."):
                streamed_output = prompt_llm_stream(prompt, uploaded_file)
                response = st.write_stream(streamed_output)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
