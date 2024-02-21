# 🎯**超强版本分享 && 多模态强化 && Chatty_Horo_Voich**🎯
> **“一段旅途的结束，是另一项征程的开始。” -> 那路**

![alt text](src/pic/Bg-Pic-1.jpeg)

## **😎"Basic Introduction" And "模型构架"😎**

**模型升级的主要贡献点如下：**

+ **进一步对 `horo2ds` 进行数据清洗，还增加了一些新内容**
+ **加入 Translation Chain(Langchain) + VITS 音频输出模式**
+ **提炼并微调了 `Horowag_7b_Chat_Ani` 模型**

![alt text](src/pic/Bg-Pic-2.png)

**该架构图展示了 `Chatty_Horo_Voich` 的核心构思。其中，我们输入的内容，也就是 `input text` 会进入 `assistant` 模块以及 `conversation chain`。这个对话链就是我们上一个视频提及的具有 `memory` 特性的架构，利用 `horowag` 模型生成赫萝的回答之后，`output text` 会在输出上稍加等待，先让 `assistant` 将输出内容置入 `translation chain` 完成中文向日语的转化，再将此日语语料加入微调后的VITS模型，输出音频。当音频输出被检测到之后，`horowag` 才会输出 output 文本。保证用户在读取模型回答的时候，同步接受音频内容。**

## **😃VITS + Finetuning + Translation Chain😃**

**VITS 是一种用于语音转化的模型结构，其配备有基座部分 `G_0.pth` 以及 `D_0.pth` 完成语音转化任务。我们可以通过调取 `config` 参数执行命令，借此在 WebUI 之中使用微调模型。有关部署建议如下：**

    pip install imageio==2.4.1
    pip install moviepy

**Build monotonic align 是非常重要的步骤，需要在 clone 后的文件夹路径下完成：**

    cd monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    cd ..

**完成此步骤后，我们就可以使用自己写好的代码完成指令控制：**

    # 定义音频构建函数
    def voice_builder(context: str):
        # 定义 API 参数
        program = "Chatty_Horo_Voich/VITS-kit/cmd_inference.py"
        api_param_args_1 = "-m" 
        api_param_conf_1 = "/home/xlab-app-center/G_10000R.pth"
        api_param_args_2 = "-c" 
        api_param_conf_2 = "/home/xlab-app-center/config.json"
        api_param_args_3 = "-o" 
        api_param_conf_3 = ""
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

![alt text](src/pic/Bg-Pic-3.png)

**Translation Chain 是一个很简单的构造，我们先加载好 `Qwen` 模型，然后根据以下代码逻辑完成 Chain 构建。其中，`system prompt` 可以根据不同角色稍作修改：**

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

## **🍟"数据集优化"🍟**

**我清洗了一下数据，保留了数据关系对话，并且将对话中模糊不清的部分加以改进。比如，以前根据语境，赫萝会回答：“怎么了？” 或者 “为什么？” 我会将其添加得更加详细，比如：怎么了？修改成 汝怎么这般狼狈模样？怎么了？   或者   为什么汝不让咱吃苹果？  尽可能减少模糊数据在训练过程中发生的冲突。**

    {
        "conversation":[
            {
                "input": "嘿，赫萝！和我聊一会儿天吧！",
                "output": "可以哦，咱和汝说，咱这两天在研究苹果派~~"
            },{
                "input": "你在研究苹果派的什么？",
                "output": "怎么挑苹果派的原料，怎么做苹果派，以及~~怎么好好享用苹果派。"
            },{
                "input": "我可以和你一起吃苹果派吗？",
                "output": "当然可以哦，不过汝要多给咱准备一些苹果才好。"
            }
        ]
    }

![alt text](src/pic/Bg-Pic-4.png)

## **🍔WebUI + Audio Output🍔**

**咱们来搞定UI细节，因为输出音频的时序逻辑，必然会使用户等待较长的时间，为了顾及体验感，我引入了 `Qwen1.5` 的 `AWQ量化` 模型，降低显存占用的同时，提升运行速度。我也写了双页面UI，让用户既可以使用生成音频的 `Voicy_Voicy` 聊天策略，也可以选取更加快速的 `Chatty_Chatty` 文本聊天策略。这样间接提升用户的体验感。**

![alt text](src/pic/Bg-Pic-5.png)

**至于为什么保留 `Chatty_Chatty`，我们以用户的角度来看，如果输出音频时间过久，那也太烦了，不如直接先进行文本对话。这样我起码能够体验模型本身的输出交互能力。因此，我把 `Chatty` 功能作为 UI 的一部分保留下来，让大家可以快速体验。当然，我也在 WebUI 下方提示之中说明了版本的不同之处。**

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


## **☃️"Bug" And "解决方案"☃️**

### **问题 1：`autoawq` 记得要控制版本**
![alt text](src/pic/Bg-Pic-6.png)

**这个包不会因为版本问题和其他环境产生冲突，实际上是发生冲突但是不会报错。如果遇到以下错误，建议降低版本 (e.g. 执行命令 `pip install autoawq==0.1.1`):**

    import awq_inference_engine  # with CUDA kernels
    ImportError: libcudart.so.12: cannot open shared object file: No such file or directory

### **问题 2：语音训练应用了巨佬的 `Google Colab` 笔记本(强推)**
**Github地址：[VITS-Fast-Fine-Tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning/tree/main)**

![alt text](src/pic/Bg-Pic-7.png)

**按照该笔记本，基本没有出现任何问题，除了再 Step.3 中需要添加内容(否则会报错)：**

    !echo '/usr/lib64-nvidia' > /etc/ld.so.conf.d/libcuda.conf; ldconfig

## **🍎"小结"🍎**

**我想专注于用更小的算力，实现更复杂的功能。我很想让这个项目一直好玩下去。接下来的目标，自然是 1.8B 的大作战啦！当然不能忘记致谢环节：感谢 InternLM 社区的算力 以及 盆友 [Claire 同学](https://space.bilibili.com/14888344?spm_id_from=333.1007.0.0) 的美术等支持！**

![alt text](src/pic/Bg-Pic-8.jpeg)

+ 书生·浦语开源仓库链接：[InternLM's tutorial](https://github.com/InternLM/tutorial/tree/main)
+ 我的项目仓库链接：[InternLM_openNotebook](https://github.com/SaaRaaS-1300/InternLM_openNotebook)
+ 我的openXLab链接：[Horowag_7b_Chat_Ani](https://openxlab.org.cn/models/detail/SaaRaaS/Horowag_7b)
