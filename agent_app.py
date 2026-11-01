import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


@st.cache_resource
def build_agent():
    # 建知识库
    with open("mydoc.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    ).split_documents([Document(page_content=text)])
    db = FAISS.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    )

    # 工具1：检索文档
    @tool
    def search_docs(query: str) -> str:
        """当需要查询项目文档中的技术细节、参数、方案说明时使用。输入应为一句具体的问题。"""
        hits = db.similarity_search(query, k=3)
        return "\n\n".join(d.page_content for d in hits)

    # 工具2：计算器
    @tool
    def calculator(expression: str) -> str:
        """用于数学计算。输入必须是合法的数学表达式，例如 (120+35)*2。"""
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"计算失败: {e}"

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    return create_react_agent(
        llm,
        tools=[search_docs, calculator],
        prompt="你是一个专业的技术文档助手。回答技术问题前，优先调用 search_docs 工具检索文档；"
               "需要计算时调用 calculator；如果文档中没有相关信息，明确说明不知道，不要编造。",
    )


st.set_page_config(page_title="文档问答 Agent", page_icon="🤖")
st.title("🤖 文档问答 Agent")
st.caption("基于 ReAct Agent + RAG + Tool Calling")

with st.spinner("正在初始化 Agent..."):
    agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考..."):
            result = agent.invoke({"messages": [{"role": "user", "content": q}]})

        # 展示 Agent 的工具调用轨迹（这是 Agent 最大的看点）
        for msg in result["messages"]:
            for tc in getattr(msg, "tool_calls", []) or []:
                st.info(f"🔧 调用工具：**{tc['name']}**　参数：`{tc['args']}`")

        answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
