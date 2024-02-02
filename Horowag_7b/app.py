from openxlab.model import download
download(model_repo='SaaRaaS/Horowag_7b',
         model_name=['pytorch_model-00001-of-00008',
                     'pytorch_model-00002-of-00008',
                     'pytorch_model-00003-of-00008',
                     'pytorch_model-00004-of-00008',
                     'pytorch_model-00005-of-00008',
                     'pytorch_model-00006-of-00008',
                     'pytorch_model-00007-of-00008',
                     'pytorch_model-00008-of-00008',
                     'config.json',
                     'configuration_internlm.py',
                     'generation_config.json',
                     'modeling_internlm2.py',
                     'pytorch_model.bin.index.json',
                     'special_tokens_map.json',
                     'tokenization_internlm.py',
                     'tokenizer.model',
                     'tokenizer_config.json'],
         output='Horowag_7b')

from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("Horowag_7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Horowag_7b", trust_remote_code=True).half()
model = model.eval()

def chat_with_model(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Generate completion
    output = model.generate(input_ids, max_length=8, num_beams=1)
    # Decode the completion
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

chat_interface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    allow_flagging="never",
)

chat_interface.launch()

