from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 加载文档
files_db_loader = DirectoryLoader("/root/RAG/file_db", show_progress=True)
files_db = files_db_loader.load()

# 向量数据库保存位置 / embedding model
model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
persist_directory = '/root/RAG/vector_db'
# 创建分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
split_docs = text_splitter.split_documents(files_db)
# 生成向量（embedding）
embedding = ModelScopeEmbeddings(model_id=model_id)

# 创建向量数据库 and 执行
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)
