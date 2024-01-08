import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/model', revision='v1.0.3')

# 下载模型
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')

from huggingface_hub import hf_hub_download  # Load model directly 
hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json")