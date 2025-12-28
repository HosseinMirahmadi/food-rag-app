import streamlit as st
import pandas as pd
import os
import shutil

# --- RAG logic libraries ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ===========================
# Page configuration
# ===========================
st.set_page_config(
    page_title="غذا و رستوران",
    page_icon="🥗",
    layout="wide"
)

# ===========================
# Style settings (CSS) for full RTL support
# ===========================
st.markdown("""
<style>
    /* 1. Font and global page direction */
    @import url('https://v1.fontapi.ir/css/Vazir');
    
    html, body, [class*="css"] {
        font-family: 'Vazir', 'Tahoma', sans-serif;
        direction: rtl;
        text-align: right;
    }

    /* 2. Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }

    /* 3. Right-align text inputs (Input & Text Area) */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    .stTextArea > div > div > textarea {
        direction: rtl;
        text-align: right;
    }

    /* 4. Card styling */
    .card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .title {
        font-size: 2.4em;
        font-weight: 800;
        color: #6ee7b7;
        text-align: right;
    }
    .subtitle {
        color: #a7f3d0;
        font-size: 1.1em;
        text-align: right;
        margin-top: 5px;
    }
    .result-text {
        color: #e2e8f0;
        font-size: 1.1em;
        line-height: 1.8;
        text-align: right;
        direction: rtl;
    }

    /* 5. Right-align tables (DataFrame) */
    [data-testid="stDataFrame"] {
        direction: rtl;
        text-align: right;
    }
    /* Attempt to right-align table headers */
    .stDataFrame div[role="columnheader"] {
        text-align: right !important;
        justify-content: right !important;
    }
    /* Table cells */
    .stDataFrame div[role="gridcell"] {
        text-align: right !important;
        justify-content: right !important;
    }
    
    /* 6. Alert and success messages */
    .stAlert {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# Application logic
# ===========================

PERSIST_DIRECTORY = "./chroma_db_food"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load and cache the embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Create and persist the knowledge base
def create_knowledge_base(urls):
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except:
            pass
            
    try:
        # Load web content
        loader = WebBaseLoader(urls)
        data = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)

        # Generate embeddings and store in Chroma
        embedding_model = load_embedding_model()
        vector_db = Chroma.from_documents(
            documents=all_splits, 
            embedding=embedding_model, 
            persist_directory=PERSIST_DIRECTORY
        )
        return True, len(all_splits)
    except Exception as e:
        return False, str(e)

# Perform RAG search and generate an answer
def perform_rag_search(query):
    embedding_model = load_embedding_model()
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    # Retrieve top-k relevant documents
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    
    # Initialize Ollama LLM
    llm = Ollama(model="llama3") 
    
    # Build context from retrieved documents
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Persian-only prompt
    prompt = f"""
    You are a professional Persian Food Expert.
    Answer the user's question strictly in PERSIAN (Farsi).
    Do NOT use English in the output.
    
    Context:
    {context_text}
    
    User Question: {query}
    
    Answer (in Persian):
    """
    
    response = llm.invoke(prompt)
    return response, docs

# ===========================
# User Interface (UI)
# ===========================

# --- Header ---
st.markdown("""
<div class="card">
    <div class="title">🥗 دستیار هوشمند غذا و رستوران</div>
    <div class="subtitle">
        جستجو در منوی رستوران‌ها، دستور پخت‌ها و مقالات غذایی
    </div>
</div>
""", unsafe_allow_html=True)

# --- Step 1: Data sources ---
st.markdown("### 🔗 مرحله ۱: منابع اطلاعاتی")
with st.container():
    input_urls = st.text_area(
        "لینک‌ها را وارد کنید (هر خط یک لینک):",
        height=100,
        placeholder="https://example.com/menu",
        value="https://fa.wikipedia.org/wiki/آشپزی_ایرانی\nhttps://fa.wikipedia.org/wiki/کباب"
    )

# --- Step 2: Processing ---
st.markdown("### 👨‍🍳 مرحله ۲: پردازش")
if st.button("🍳 بررسی و یادگیری"):
    if input_urls.strip():
        url_list = [u.strip() for u in input_urls.split('\n') if u.strip()]
        with st.spinner('در حال خواندن منوها...'):
            success, result = create_knowledge_base(url_list)

        if success:
            st.success(f"✅ انجام شد! {result} بخش متنی ذخیره شد.")
            st.session_state["db_ready"] = True
        else:
            st.error(f"❌ خطا: {result}")
    else:
        st.warning("⚠️ لطفاً لینک وارد کنید.")

# --- Step 3: Q&A ---
if st.session_state.get("db_ready"):
    st.markdown("### 🍽️ مرحله ۳: پرسش و پاسخ")

    # In RTL layout, the first column appears on the right
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "سوال شما:",
            placeholder="مثلاً: کباب کوبیده خوب چه ویژگی‌هایی دارد؟"
        )
    with col2:
        # Vertical alignment for the search button
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        search = st.button("🔎 جستجو", use_container_width=True)

    if search and query:
        with st.spinner('در حال نوشتن پاسخ...'):
            try:
                ai_response, source_docs = perform_rag_search(query)
                
                # Display AI answer
                st.markdown(f"""
                <div class="card">
                    <h3 style="color:#fbbf24; text-align:right; margin-bottom:10px;">
                        🍕 پاسخ هوش مصنوعی:
                    </h3>
                    <div class="result-text">
                    {ai_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display sources table
                st.markdown("### 📜 منابع یافت شده")
                
                table_data = []
                for idx, doc in enumerate(source_docs):
                    table_data.append({
                        "رتبه": idx + 1,
                        "متن (خلاصه)": doc.page_content[:150] + "...",
                        "لینک منبع": doc.metadata.get('source', 'نامشخص'),
                    })
                
                df = pd.DataFrame(table_data)
                
                # Render DataFrame with RTL-friendly settings
                st.dataframe(
                    df, 
                    use_container_width=True,
                    column_config={
                        "لینک منبع": st.column_config.LinkColumn("لینک کامل"),
                        "رتبه": st.column_config.NumberColumn("رتبه", format="%d")
                    },
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"خطا: {e}")
