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
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, repetition_penalty=1.002).to(torch.bfloat16).cuda()
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
        system_prompt = ""
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


"""
    推荐使用 InternLM2-Chat-1_8b 及其微调模型
"""
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
        system_prompt = ""
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
