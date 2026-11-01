import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 读文档（直接用 Python 内置方法读，绕开 community 包）
with open("mydoc.txt", "r", encoding="utf-8") as f:
    text = f.read()
docs = [Document(page_content=text, metadata={"source": "mydoc.txt"})]
print("文档加载完成")

# 2. 切分
chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
).split_documents(docs)
print(f"切分成 {len(chunks)} 个片段")

# 3. 向量化（首次运行会下载约 100MB 模型）
print("正在加载向量模型，首次运行需要下载，请耐心等待...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = FAISS.from_documents(chunks, embeddings)
print("知识库构建完成！\n")

# 4. 问答
llm = ChatOpenAI(model="deepseek-chat",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com")

while True:
    q = input("你的问题（输入 q 退出）：")
    if q.lower() == "q":
        break
    related = db.similarity_search(q, k=3)
    context = "\n\n".join(d.page_content for d in related)
    resp = llm.invoke(
        f"仅根据以下资料回答问题，资料中没有的信息就说不知道：\n\n{context}\n\n问题：{q}"
    )
    print("\n回答：", resp.content, "\n")
