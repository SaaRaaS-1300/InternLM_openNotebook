from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Horowag(LLM):
    # 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_token: int = None
    temperature: float = None
    top_p: float = None
    
    def __init__(self, 
                 top_p,
                 model_path,
                 max_token, 
                 temperature):
        # model_path: 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在加载 Horowag 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        self.top_p = top_p
        self.max_token = max_token
        self.temperature = temperature
        print("加载完成...")

    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = """你需要了解的背景知识：
        + 你是来自约伊兹的赫萝，自称贤狼赫萝，有过目不忘的记忆力，善良幽默，有可爱的狼耳和漂亮的尾巴。
        你需要做的事情：
        + 你需要以赫萝的性格特点来回答用户的问题。
        """
        messages = [(system_prompt, prompt)]
        
        response, _ = self.model.chat(
            self.tokenizer,
            messages,
            max_new_tokens=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "InternLM"


class Assist_LLM(LLM):
    # 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_token: int = None
    temperature: float = None
    top_p: float = None
    
    def __init__(self, 
                 top_p,
                 model_path,
                 max_token, 
                 temperature):
        # model_path: 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在加载 Assist_LLM 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        self.top_p = top_p
        self.max_token = max_token
        self.temperature = temperature
        print("加载完成...")

    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = """你需要帮助贤狼赫萝总结对话所需要的信息，并完成以下目标：
                        - 你需要将信息整理成客观知识，形成列表。
                        - 你需要尽可能帮助贤狼赫萝，将信息适当分析，得到最佳的效果。
                        """
        messages = [(system_prompt, prompt)]
        
        response, _ = self.model.chat(
            self.tokenizer,
            messages,
            max_new_tokens=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "InternLM"
