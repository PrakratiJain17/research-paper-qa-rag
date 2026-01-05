# 🔬 Research Paper QA System using RAG

An end-to-end **Research Paper Question Answering System** built using **Retrieval-Augmented Generation (RAG)**. This application allows users to upload one or more research papers (PDFs) and ask natural language questions to get accurate, context-aware answers grounded strictly in the uploaded documents.

Built with **Streamlit**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **GROQ (LLaMA 3.3 – 70B)**.

---

## 🚀 Features

* 📄 Upload **multiple PDF research papers**
* 🔍 Semantic search using **FAISS vector database**
* 🧠 Context-aware answers using **RAG pipeline**
* 🤖 Powerful LLM: **LLaMA 3.3 (70B) via GROQ**
* 📌 Source attribution (page number + document)
* 💬 Interactive Q&A chat history
* 🎨 Custom UI with Streamlit + TOML theming
* ⚡ Cached models for faster performance

---

## 📸 Application Interface

### 📄 PDF Upload & Processing

![PDF Upload Interface](assets/interface (1).png)

### 💬 Question Answering Interface

![QA Interface](assets/interface (2).png)

---

## 🛠️ Tech Stack

| Component     | Technology                     |
| ------------- | ------------------------------ |
| Frontend      | Streamlit                      |
| LLM           | LLaMA 3.3 (70B) – GROQ         |
| RAG Framework | LangChain                      |
| Embeddings    | sentence-transformers (MiniLM) |
| Vector Store  | FAISS                          |
| PDF Parsing   | PyPDF                          |
| Environment   | Python, Virtualenv             |

---

## 📁 Project Structure

```
research-paper-qa-rag/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # GROQ API key (not committed)
├── .streamlit/
│   └── config.toml         # Theme & server configuration
├── README.md               # Project documentation
└── venv/                   # Virtual environment
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PrakratiJain17/research-paper-qa-rag.git
cd RAG_QA

```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If FAISS causes issues on Windows:

```bash
pip install faiss-cpu --no-cache-dir
```

---

### 4️⃣ Set Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```


---

### 5️⃣ (Optional) Streamlit Theme Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

---

### 6️⃣ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 💡 Example Questions

* What is the main contribution of this paper?
* What methodology is used?
* What datasets were used for evaluation?
* What are the key findings?
* What limitations are mentioned?
* How does this compare with previous work?

---

## 🔐 Security Notes

* `.env` file is excluded via `.gitignore`
* API keys are never hardcoded
* Vector DB is created in-memory per session

---

## 📌 Use Cases

* Literature review automation
* Research assistance for students
* Paper understanding for interviews
* Academic project demonstrations
* NLP / RAG portfolio project

---

## 🧩 Future Enhancements

* Persistent vector database
* PDF page highlighting
* Multi-LLM selection
* Authentication & user sessions
* Cloud deployment (AWS / HuggingFace Spaces)

---

## 👩‍💻 Author

**Prakrati Jain**



---

## ⭐ Acknowledgements

* LangChain
* GROQ
* HuggingFace
* Streamlit

---
