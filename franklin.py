import streamlit as st
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    st.warning(f"No text found on page {pdf_reader.pages.index(page)+1} of {pdf.name}")
        except PdfReadError:
            st.warning(f"Could not read PDF: {pdf.name}. It may be corrupted or invalid.")
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create vector store (FAISS) from chunks and save locally
def create_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks found, upload readable PDFs!")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load QA chain using Gemini chat
def get_qa_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-bro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Query user input question and show answer
def answer_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"input_documents": docs, "question": question})
    st.write("### Answer:")
    st.write(response["output_text"])

def main():
    st.set_page_config(page_title="PDF Chat with Google Gemini Embeddings", layout="wide")
    st.title("📄 Welcome to Techspark PDF chatbat")
    
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            with st.spinner("Extracting and embedding PDF text..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No readable text found in PDFs. If scanned, consider OCR!")
                else:
                    chunks = get_text_chunks(raw_text)
                    create_vector_store(chunks)
                    st.success("Vector store created successfully!")

    user_question = st.text_input("Ask a question about the PDFs:")
    if user_question:
        answer_question(user_question)

if __name__ == "__main__":
    main()
