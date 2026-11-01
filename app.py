import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="文档问答助手", page_icon="🤖")
st.title("🤖 文档问答助手")
st.caption("基于 RAG（检索增强生成）+ DeepSeek 大模型")


# 缓存：只在第一次加载时构建，避免每次提问都重建索引
@st.cache_resource
def build_db():
    with open("mydoc.txt", "r", encoding="utf-8") as f:
        text = f.read()
    docs = [Document(page_content=text, metadata={"source": "mydoc.txt"})]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


with st.spinner("正在加载知识库，请稍候..."):
    db = build_db()
llm = get_llm()

# 会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 输入框
if q := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    related = db.similarity_search(q, k=3)
    context = "\n\n".join(d.page_content for d in related)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            resp = llm.invoke(
                f"仅根据以下资料回答问题，资料中没有的信息就说不知道：\n\n{context}\n\n问题：{q}"
            )
            st.markdown(resp.content)

        # 展示检索到的原始资料（可折叠）——面试时是个很好的展示点
        with st.expander("查看检索到的参考片段"):
            for i, d in enumerate(related, 1):
                st.text(f"[片段{i}] {d.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": resp.content})
