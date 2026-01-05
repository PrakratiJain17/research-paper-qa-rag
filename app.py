"""
Research Paper QA System with RAG - Streamlit Interface
Author: Prakrati Jain
GitHub: github.com/yourusername/research-paper-qa-rag
"""

import os
import torch
import streamlit as st
from dotenv import load_dotenv
import pypdf
import tempfile

# LangChain imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Research Paper QA System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """Initialize models (cached to avoid reloading)"""
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found! Please set it in .env file")
        st.stop()
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
        api_key=api_key
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    return llm, embeddings, device


def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)


def load_and_process_pdf(pdf_file):
    """Load PDF and split into chunks"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return chunks, len(documents)


def create_rag_chain(chunks, llm, embeddings):
    """Create RAG chain"""
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Prompt template
    prompt_template = """You are a helpful research assistant. Use the following context to answer the question.
If you don't know the answer, say "I cannot find this information in the provided papers."
Always mention relevant details from the context.

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize models
llm, embeddings, device = initialize_models()

# Header
st.title("🔬 Research Paper QA System")
st.markdown("### Ask questions about your research papers using RAG")

# Sidebar
with st.sidebar:
    st.header("📄 Upload Papers")
    
    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers"
    )
    
    if st.button("🚀 Process Papers", type="primary"):
        if uploaded_files:
            with st.spinner("Processing papers..."):
                try:
                    all_chunks = []
                    total_pages = 0
                    file_names = []
                    
                    # Process each PDF
                    progress_bar = st.progress(0)
                    for idx, pdf_file in enumerate(uploaded_files):
                        st.write(f"📄 Processing: {pdf_file.name}")
                        chunks, pages = load_and_process_pdf(pdf_file)
                        all_chunks.extend(chunks)
                        total_pages += pages
                        file_names.append(pdf_file.name)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Create RAG chain
                    st.write("🔍 Building vector database...")
                    rag_chain, retriever = create_rag_chain(all_chunks, llm, embeddings)
                    
                    # Store in session state
                    st.session_state.rag_chain = rag_chain
                    st.session_state.retriever = retriever
                    st.session_state.processed_files = file_names
                    
                    st.success(f"""
                    ✅ **Success!**
                    - Papers: {len(uploaded_files)}
                    - Pages: {total_pages}
                    - Chunks: {len(all_chunks)}
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please upload at least one PDF file")
    
    # Show processed files
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📚 Processed Papers")
        for file in st.session_state.processed_files:
            st.write(f"✓ {file}")
    
    st.divider()
    
    # System info
    st.subheader("⚙️ System Info")
    st.write(f"🎮 Device: {device.upper()}")
    st.write(f"🤖 Model: Llama-3.3-70B")
    st.write(f"🔍 Vector DB: FAISS")
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the main contribution of this paper?",
        height=100,
        key="question_input"
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Process question
    if ask_button and question:
        if st.session_state.rag_chain is None:
            st.error("❌ Please upload and process papers first!")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Get answer
                    answer = st.session_state.rag_chain.invoke(question)
                    
                    # Get sources
                    source_docs = st.session_state.retriever.invoke(question)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'sources': source_docs
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display chat history (most recent first)
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📝 Q&A History")
        
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history) - idx}:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                
                # Sources
                if chat['sources']:
                    with st.expander("📌 View Sources"):
                        for i, doc in enumerate(chat['sources'][:3], 1):
                            page = doc.metadata.get('page', 'N/A')
                            source = doc.metadata.get('source', 'N/A')
                            st.write(f"{i}. Page {int(page) + 1} from {source.split('/')[-1]}")
                            st.caption(doc.page_content[:200] + "...")
                
                st.divider()

with col2:
    st.header("💡 Example Questions")
    
    examples = [
        "What is the main contribution of this paper?",
        "What methodology was used in this research?",
        "What are the key findings and results?",
        "What datasets were used for evaluation?",
        "What are the performance metrics reported?",
        "What are the limitations of this work?",
        "How does this compare to previous work?",
        "What future work is suggested?"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example[:20]}"):
            st.session_state.question_input = example
            st.rerun()
    
    st.divider()
    
    st.subheader("🎯 Tips for Better Results")
    st.markdown("""
    - Be specific in your questions
    - Ask about concrete details
    - Reference specific concepts from papers
    - Try rephrasing if answer isn't satisfactory
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using LangChain, GROQ, and Streamlit | 
    <a href='https://github.com/yourusername/research-paper-qa-rag'>GitHub</a>
</div>
""", unsafe_allow_html=True)