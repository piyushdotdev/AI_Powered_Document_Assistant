import streamlit as st
import os
import asyncio
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ and OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title and description
st.set_page_config(page_title="AI Powered Q&A", layout="wide")
st.title("📄 AI-Powered Interactive Q&A System")
st.subheader("Quickly analyze PDFs and get answers from your documents in real-time.")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Initialize session state variables
if "documents" not in st.session_state:
    st.session_state.documents = []  # List to store document texts
    st.session_state.vectors = None  # Vector store

# Function to process uploaded PDF files asynchronously
async def process_uploaded_files(uploaded_files):
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Progress bar for file processing
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for index, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file temporarily to disk
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process each PDF
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()
        st.session_state.documents.extend(docs)  # Add to session state
        new_documents.extend(text_splitter.split_documents(docs))  # Create chunks
        
        # Remove the temporary file after processing
        os.remove(uploaded_file.name)
        
        # Update the progress bar
        progress_bar.progress((index + 1) / total_files)
    
    # Update or create a vector store
    if st.session_state.vectors is None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(new_documents, st.session_state.embeddings)
    else:
        # Add new documents to the existing vector store
        st.session_state.vectors.add_documents(new_documents)
    
    # Complete the progress bar
    progress_bar.empty()
    st.success("Documents processed and added to vector store.")

# File uploader for uploading PDFs
with st.sidebar:
    st.subheader("Upload your PDF documents:")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    # Button to trigger document processing
    if st.button("Process Uploaded Documents") and uploaded_files:
        asyncio.run(process_uploaded_files(uploaded_files))
        st.sidebar.success("All PDFs processed!")

# Input for user to ask questions
st.subheader("Ask Questions from Your Documents:")
prompt1 = st.text_input("Enter your question")

# Response generation with progress indicator
if prompt1:
    if st.session_state.vectors:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        with st.spinner("Fetching the answer..."):
            # Measure response time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            response_time = time.process_time() - start
            
        st.write("⏳ Response time: {:.2f} seconds".format(response_time))
        st.write("### Answer: ", response['answer'])
        
        # Display relevant document chunks in an expander
        with st.expander("Relevant Documents"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(f"**Document {i+1}:** {doc.page_content}")
                st.write("---")
    else:
        st.error("Please process documents before asking questions.")

# Better Layout
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .css-1aumxhk {
            background-color: #fafafa;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)
