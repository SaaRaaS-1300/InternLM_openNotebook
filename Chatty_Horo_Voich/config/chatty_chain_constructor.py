from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,  
    SystemMessagePromptTemplate,  
    HumanMessagePromptTemplate  
)
import os

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

# 构建翻译链
def qwen_translation_chain(llm):
    '''
        Langchain(Chat) + Qwen(Translation)
    '''
    # system + human
    template = """你是一个可靠的翻译专家。
    - 你需要帮助赫萝把{source_language}翻译成{target_language}。
    - 你需要模仿赫萝的语气。
    - '咱'翻译成'わっち'。
    - '汝'翻译成'ぬし'。
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)  
    # 待翻译文本由 Human 输入  
    human_template = "{text}"  
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)  
    # System + Human 提示模板 ChatPromptTemplate  
    chat_prompt_template = ChatPromptTemplate.from_messages(  
        [system_message_prompt, human_message_prompt]  
    )

    translation_chain = LLMChain(
        llm=llm, 
        prompt=chat_prompt_template, 
    )
    
    return translation_chain
