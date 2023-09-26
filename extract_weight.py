import torch
from transformers import AutoModelForCausalLM

ckpt_path = "out/llama_7b/unstructured/wanda/seed0/model"
out_path = "out/weight_0.pt"


model = AutoModelForCausalLM.from_pretrained(
    ckpt_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

weight = model.model.layers[0].self_attn.q_proj.weight
torch.save(weight, out_path)