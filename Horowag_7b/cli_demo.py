"""
    使用 test_tinyLlama 验证代码的可行性
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained('model/test_tinyLlama/', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("model/test_tinyLlama/", trust_remote_code=True, padding_side="left")

prompt = "Hello, world"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate completion
output = model.generate(input_ids, max_length=8, num_beams=1)
# Decode the completion
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(output_text)
