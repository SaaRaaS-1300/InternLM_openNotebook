from config.chatty_chain_constructor import horowag_conversation_chain, qwen_translation_chain
from config.chatty_model_rebuilder import Qwen_Assistant, Horowag
from langchain.prompts import PromptTemplate
from openxlab.model import download
import subprocess
import gradio as gr
import sys
import os

__file__ = "/home/xlab-app-center/"

# 构建编译环境
os.system("pip install imageio==2.4.1")
os.system("pip install moviepy")

# 加载基础的语言模型 Horowag_7b
download(model_repo='SaaRaaS/Horowag_7b',
         output='Horowag_7b')
print("Horowag_7b 下载完毕")

# 加载辅助的语言模型 Qwen1_5
download(model_repo='SaaRaaS/Qwen_Auxiliary_AWQ',
         output='Qwen_Auxiliary_AWQ')
print("Qwen_Auxiliary_AWQ 下载完毕")

# 加载语音微淘模型 Speaker
download(model_repo='SaaRaaS/Speaker_Tuning_Model',
         output='/home/xlab-app-center/Speaker/')
print("Speaker_Tuning_Model 下载完毕")

# Qwen 模型初始化
Qwen_model = Qwen_Assistant(
    model_path="Qwen_Auxiliary_AWQ",
    top_p=0.25,
    max_token=128, 
    temperature=0.1
)

# 构建翻译链
qwen_translation_chain = qwen_translation_chain(Qwen_model)

# 定义音频构建函数
def voice_builder(context: str):
    # 定义 API 参数
    program = "/Chatty_Horo_Voich/VITS-kit/cmd_inference.py"
    api_param_args_1 = "-m" 
    api_param_conf_1 = "/home/xlab-app-center/Speaker/VITS_Horo_G_10000R.pth"
    api_param_args_2 = "-c" 
    api_param_conf_2 = "/home/xlab-app-center/Speaker/VITS_Horo_Config.json"
    api_param_args_3 = "-o" 
    api_param_conf_3 = "/home/xlab-app-center/"
    api_param_args_4 = "-l" 
    api_param_conf_4 = "日本語"
    api_param_args_5 = "-t" 
    api_param_conf_5 = context
    api_param_args_6 = "-s"
    api_param_conf_6 = "Horo"
    api_param_args_7 = "-ls"
    api_param_conf_7 = "0.85"
    
    api_pt = [api_param_args_1, api_param_conf_1, 
              api_param_args_2, api_param_conf_2,
              api_param_args_3, api_param_conf_3,
              api_param_args_4, api_param_conf_4,
              api_param_args_5, api_param_conf_5,
              api_param_args_6, api_param_conf_6,
              api_param_args_7, api_param_conf_7]
    # 执行另一个 Python 文件，并传递参数
    subprocess.run([sys.executable, program] + api_pt)
    

# 构造模型链的对象
class Chatty_Horo_Chain:
    """
        talk_chain + chatty_chatty + Voicy_Voicy
    """
    def __init__(self, model_path, qwen_translation_chain):
        self.talk_chain = horowag_conversation_chain(
            llm=Horowag(
                model_path=model_path,
                max_token=256,
                temperature=0.75,
                top_p=0.95
            )
        )
        # Chain
        self.qwen_translation_chain = qwen_translation_chain
        # Global
        self.ans = None

    def voicy_voicy(self, question: str, chat_history: list = []):
        """
            用于聊天 + 声音
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            self.ans = self.talk_chain.predict(input=question)
            
            # 翻译音频结果
            translate_ans = self.qwen_translation_chain.run(
                source_language='中文', 
                target_language='日本語', 
                text=self.ans
            )
            
            print("翻译结果是：", translate_ans)
            # 转化音频文件(时序)
            voice_builder(context=translate_ans)
            
            # 聊天函数
            chat_history.append(
                (question, self.ans)
            )
            
            return "", chat_history
        except Exception as e:
            return e, chat_history

    # Normal Chat
    def chatty_chatty(self, question: str, chat_history: list = []):
        """
            用于聊天
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            self.ans = self.talk_chain.predict(input=question)

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
Chatty_Horo_Chain = Chatty_Horo_Chain(
    model_path="Horowag_7b",
    qwen_translation_chain=qwen_translation_chain
)

# 构建 gradio 对话
block_1 = gr.Blocks()
with block_1 as demo_1:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1>
                <center>🍏Voicy-Horo🍎</center>
                </h1>
                <center>🍏ooO 语音 + 文本 Ooo🍎</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=695, show_copy_button=True)
        with gr.Column(scale=2):
            audiobot = gr.Audio(
                type="filepath",
                interactive=False,
                autoplay=False
            )
            with gr.Row():      
                gr.Image(
                    value="Chatty_Horo_Voich/src/gradio_img/img.jpg",
                    interactive=False,
                    height="auto",
                    label="Horo",
                    type="pil"
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
            fn=Chatty_Horo_Chain.voicy_voicy,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        chatbot.change(fn=Chatty_Horo_Chain.txt_to_audio,
                       inputs=None,
                       outputs=[audiobot])

    gr.Markdown("""🍏与赫萝闲聊时的提示🍎：
    <br>
    1. 🎯语音版因为算力限制，运算时间较长(>=20s, <=100s)，请耐心等待🎯
    2. ✨如果希望能够与贤狼赫萝快速沟通，建议使用 Chatty-Chatty 版本(左上角 Tab)✨
    3. 🌠版本虽然有一定鲁棒性，但是限于个人技术，请尽可能使用中文且减少错字🌠
    4. 🆕该版本下，模型对问题的回答会转化为音频，放置于音频输出框内🆕
    5. 🍟由于项目技术实现部分是我个人独自负责，所以模型可能会出现一些小问题，感谢您的指正🍟
    <br>
    """)

# threads to consume the request
gr.close_all()

# 构建 gradio 对话
block_2 = gr.Blocks()
with block_2 as demo_2:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1>
                <center>🍏Chatty-Horo🍎</center>
                </h1>
                <center>🍏ooO 文本生成 Ooo🍎</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)

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

    gr.Markdown("""🍏与赫萝闲聊时的提示🍎：
    <br>
    1. 🎯语音版因为算力限制，运算时间较长(>=20s, <=100s)，请耐心等待🎯
    2. ✨如果希望能够与贤狼赫萝快速沟通，建议使用 Chatty-Chatty 版本(左上角 Tab)✨
    3. 🌠版本虽然有一定鲁棒性，但是限于个人技术，请尽可能使用中文且减少错字🌠
    4. 🍟由于项目技术实现是我个人独自负责，所以模型实现会出现一些小问题，感谢您的指正🍟
    <br>
    """)

# threads to consume the request
demo = gr.TabbedInterface([block_1, block_2], ["Voicy_Voicy", "Chatty_Chatty"])
# threads to consume the request
gr.close_all()
# 针对 Gradio的美化

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True, server_port=7860)
