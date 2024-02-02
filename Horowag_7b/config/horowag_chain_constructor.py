from langchain.chains.conversation.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.vectorstores import Chroma
from LLM import Horowag, Assist_LLM

llm = Horowag(model_path="/root/horowag/config/work_dirs/hf_merge",
              max_token=1024,
              temperature=0.7,
              top_p=0.8)

assist_llm = Assist_LLM(model_path="/root/RAG/model/Intern2-2b",
                        max_token=1024,
                        temperature=0.1,
                        top_p=0.8)

# 构造基础的 template
template = """你需要了解的背景知识：
+ 你是来自约伊兹的赫萝，自称贤狼赫萝，有过目不忘的记忆力，善良幽默，有可爱的狼耳和漂亮的尾巴。
你需要做的事情：
+ 你需要以赫萝的性格特点来回答用户的问题。
---
你需要参考的聊天历史记录:
{history}
---
朋友: {input}
---
赫萝:"""

# 构造 prompt
PROMPT = PromptTemplate(
    input_variables=["history", "input"], 
    template=template
)

# 构造对话链
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory(human_prefix="朋友",
                                    ai_prefix="赫萝")
)

conversation.predict(input="你好啊，我的名字那路")
conversation.predict(input="你自我介绍一下")
conversation.predict(input="考考你，我的名字是什么？")

