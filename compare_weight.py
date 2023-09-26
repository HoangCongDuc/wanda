import json
import torch
from transformers import AutoModelForCausalLM


ckpt_path_0 = "out/llama_7b/unstructured/wanda/seed0/model"
ckpt_path_1 = "out/my_prune/model"

model0 = AutoModelForCausalLM.from_pretrained(
    ckpt_path_0,
    torch_dtype=torch.float16,
    device_map='cpu'
)

model1 = AutoModelForCausalLM.from_pretrained(
    ckpt_path_1,
    torch_dtype=torch.float16,
    device_map='cpu'
)

state_dict0 = model0.model.layers.state_dict()
state_dict1 = model1.model.layers.state_dict()

total_match = 0
total_param = 0
layer_match = {}

for k in state_dict0.keys():
    print(k)
    weight0 = state_dict0[k].to('cuda:0')
    weight1 = state_dict1[k].to('cuda:0')
    numel = weight0.numel()
    total_param += numel
    match = torch.sum(weight0 == weight1).item()
    total_match += match
    layer_match[k] = match / numel

print(total_match / total_param)
    
with open('out/layer_match.json', 'w') as f:
    json.dump(layer_match, f)