from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


# 构建基础对话链
def horowag_conversation_chain(llm):
    '''
        Langchain(Conversation) + Horowag
    '''
    # 构造基础的 talk_template
    talk_template = """你需要了解的背景知识：
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
        template=talk_template
    )

    # 构造记忆链
    memory = ConversationBufferMemory(
        human_prefix="朋友",
        ai_prefix="赫萝"
    )

    # 构造对话链
    talk_chain = ConversationChain(
        llm=llm, 
        prompt=PROMPT,
        memory=memory,
    )

    return talk_chain
