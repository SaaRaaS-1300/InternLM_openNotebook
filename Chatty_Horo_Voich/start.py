from config.chatty_chain_constructor import horowag_conversation_chain
from config.chatty_model_rebuilder import Horowag
import subprocess
import gradio as gr
import sys
import os

__file__ = "/root/Chatty_Horo_Voich/Voice/"


# 构造模型链的对象
class Chatty_Horo_Chain():
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
        # Global
        self.ans = None

    def chatty_chatty(self, question: str, chat_history: list = []):
        """
            用于聊天
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            self.ans = self.talk_chain.predict(input=question)
            # 转化音频文件(时序)
            
            # 定义 API 参数
            program = "/root/Chatty_Horo_Voich/VITS-fast-fine-tuning/cmd_inference.py"
            api_param_args_1 = "-m" 
            api_param_conf_1 = "/root/Chatty_Horo_Voich/V_Model/Test/module/G_4500.pth"
            api_param_args_2 = "-c" 
            api_param_conf_2 = "/root/Chatty_Horo_Voich/V_Model/Test/config/config.json"
            api_param_args_3 = "-o" 
            api_param_conf_3 = "/root/Chatty_Horo_Voich/Voice/"
            api_param_args_4 = "-l" 
            api_param_conf_4 = "简体中文"
            api_param_args_5 = "-t" 
            api_param_conf_5 = self.ans
            api_param_args_6 = "-s"
            api_param_conf_6 = "hiyori"
            api_param_args_7 = "-ls"
            api_param_conf_7 = "0.9"
            
            api_pt = [api_param_args_1, api_param_conf_1, 
                      api_param_args_2, api_param_conf_2,
                      api_param_args_3, api_param_conf_3,
                      api_param_args_4, api_param_conf_4,
                      api_param_args_5, api_param_conf_5,
                      api_param_args_6, api_param_conf_6,
                      api_param_args_7, api_param_conf_7]

            print(api_pt)
            # 执行另一个 Python 文件，并传递参数
            subprocess.run([sys.executable, program] + api_pt)
            
            # 聊天函数
            chat_history.append(
                (question, self.ans)
            )
            
            return "", chat_history
        except Exception as e:
            return e, chat_history

    # 音频模块
    def txt_to_audio(self):
        """
            用于音频转化
        """
        # 路径
        return os.path.join(os.path.dirname(__file__), "output.wav")


# 构建对话模式
Chatty_Horo_Chain = Chatty_Horo_Chain(model_path="/root/horowag/config/work_dirs/hf_merge")

# 构建 gradio 对话
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1><center>Horowag-7b-V2</center></h1>
                <center>🍏与赫萝闲聊🍎</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
        with gr.Column(scale=2):
            audiobot = gr.Audio(
                type="filepath",
                interactive=False,
                autoplay=False
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
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
            fn=Chatty_Horo_Chain.chatty_chatty,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        chatbot.change(fn=Chatty_Horo_Chain.txt_to_audio,
                       inputs=None,
                       outputs=[audiobot])

    gr.Markdown("""提醒：
    <br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 
    <br>
    """)

# threads to consume the request
gr.close_all()
# 针对 Gradio的美化

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True)
