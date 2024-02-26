from config.mini_chain_constructor import horowag_conversation_chain
from config.mini_model_rebuilder import Horowag
from openxlab.model import download
import gradio as gr

# theme 约定
theme = 'ysharma/llamas'

# 加载基础的语言模型 Horowag_Mini
download(model_repo='SaaRaaS/Horowag_Mini',
         output='Horowag_Mini')


# 构造模型链的对象
class Horo_Chatty_Chain():
    """
        talk_chain + chatty_chatty
    """
    def __init__(self, model_path):
        self.talk_chain = horowag_conversation_chain(
            llm=Horowag(
                model_path=model_path,
                max_token=256,
                temperature=0.75,
                top_p=0.95
            )
        )

    def chatty_chatty(self, question: str, chat_history: list = []):
        """
            用于聊天的函数
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.talk_chain.predict(input=question)))
            return "", chat_history
        except Exception as e:
            return e, chat_history


# 构建对话模式
Horo_Chatty_Chain = Horo_Chatty_Chain(model_path='Horowag_Mini')

# 构建 gradio 对话
block = gr.Blocks(theme=theme)
with block as demo:
    with gr.Row(equal_height=True): 
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <center>🍏Chatty-Horo-Mini🍎</center>
                """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="在此输入聊天内容...")

            with gr.Row():
                # 创建提交按钮。
                chatty_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        chatty_btn.click(
            fn=Horo_Chatty_Chain.chatty_chatty, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot]
        )
        
    gr.Markdown("""🍏与赫萝闲聊时的提示🍎：
    <br>
    1. 😃此版本基于 InternLM2-Chat-1.8b 构建😃
    2. ✨由于参数量较小，该版本的最大优势在于部署成本低，而非运行效果优越。(相较于 7b 模型)✨ 
    <br>
    """)

# threads to consume the request
gr.close_all()

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True, server_port=7860)
