from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from lib.data import get_code, get_loaders
from lib.layerwrapper import WrappedLayer


def get_attention_mask(attention_mask_bool, device):
    batch_size = attention_mask_bool.shape[0]
    sequence_length = attention_mask_bool.shape[1]
    attention_mask = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.float16)
    indices = torch.triu_indices(sequence_length, sequence_length, offset=1)
    attention_mask[indices[0], indices[1]] = -65504
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
    for i in range(batch_size):
        actual_seq_length = attention_mask_bool[i].sum()
        attention_mask[i, :, :, actual_seq_length:] = -65504
    return attention_mask

torch.set_grad_enabled(False)
parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, default="~/models/models--Phind--Phind-CodeLlama-34B-v2")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--nsamples', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sparsity_ratio', type=float, default=0.5)
parser.add_argument('--data', type=str, default='code')
args = parser.parse_args()

model_dir = Path(args.model_dir).expanduser()
nsamples = args.nsamples
seed = args.seed
batch_size = args.batch_size
sparsity_ratio = args.sparsity_ratio
execution_device = 'cuda:0'
print(args)


# config = AutoConfig.from_pretrained(model_dir)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map='cpu'
)

# device_map = {}
# device_map['model.embed_tokens'] = 0
# for i in range(len(model.model.layers)):
#     device_map[f'model.layers.{i}'] = 'cpu'
# device_map['model.norm'] = 'cpu'
# device_map['lm_head'] = 'cpu'

# model = load_checkpoint_and_dispatch(
#     model,
#     checkpoint= model_dir,
#     device_map=device_map,
#     dtype=torch.float16,
#     no_split_module_classes=['LlamaDecoderLayer']
# )

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=(args.data == 'code'))

if args.data == 'code':
    train_samples = get_code(nsamples, seed, tokenizer)
else:
    trainloader, valenc = get_loaders(args.data, nsamples, seed, 2048, tokenizer)
    train_samples = [sample[0] for sample in trainloader]
    train_samples = torch.cat(train_samples, dim=0)
    train_samples = {'input_ids': train_samples}


hidden_states = model.get_input_embeddings()(train_samples['input_ids'])
sequence_length = hidden_states.shape[1]
position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
attention_mask = get_attention_mask(attention_mask, execution_device)

for i, layer in enumerate(model.model.layers):
    print(f"Pruining layer {i}")
    pruned_layers = {}
    wrapped_layers = {}
    handles = []
    # if hasattr(layer, '_hf_hook'):
    #     execution_device = layer._hf_hook.execution_device
    # else:
    #     weight = next(iter(layer.parameters()))
    #     execution_device = weight.device
    temp_layer = layer.to(execution_device)
    hidden_states = hidden_states.to(execution_device)
    position_ids = position_ids.to(execution_device)
    for name, module in temp_layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        pruned_layers[name] = module

        wrapped_layer = WrappedLayer(module, device=execution_device)
        wrapped_layers[name] = wrapped_layer

        hook = lambda m, inp, out, name=name: wrapped_layers[name].add_batch(inp[0], out)
        handle = module.register_forward_hook(hook)
        handles.append(handle)
    
    start_index = 0
    while start_index < nsamples:
        attention_mask = get_attention_mask(train_samples['attention_mask'][start_index:start_index+batch_size], execution_device)
        temp_layer(hidden_states[start_index:start_index+batch_size],
              attention_mask=attention_mask,
              position_ids=position_ids
        )
        start_index += batch_size
    for handle in handles:
        handle.remove()
        
    for name, module in pruned_layers.items():
        scaler = torch.sqrt(wrapped_layers[name].scaler_row).to(execution_device)
        weight = module.weight
        # if weight.device.type == 'meta':
        #     weight = module._hf_hook.weights_map['weight']
        W_importance = torch.abs(weight.data) * scaler
        W_mask = torch.full_like(W_importance, False, dtype=torch.bool, device=execution_device)
        sort_res = torch.argsort(W_importance, dim=1, stable=True)
        prune_indices = sort_res[:, :int(module.weight.data.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, prune_indices, True)
        weight.data[W_mask] = 0
        
    outs = []
    start_index = 0
    while start_index < nsamples:
        attention_mask = get_attention_mask(train_samples['attention_mask'][start_index:start_index+batch_size], execution_device)
        out = temp_layer(hidden_states[start_index:start_index+batch_size],
                    attention_mask=attention_mask,
                    position_ids=position_ids
        )
        outs.append(out[0])
        start_index += batch_size
    hidden_states = torch.cat(outs, dim=0)
    layer.load_state_dict(temp_layer.cpu().state_dict())
    torch.cuda.empty_cache()

def get_weight_map(module):
    if hasattr(module, '_hf_hook') and module._hf_hook.weights_map is not None:
        return module._hf_hook.weights_map.dataset
    for submodule in module.children():
        weight_map = get_weight_map(submodule)
        if weight_map is not None:
            return weight_map
    
print("Saving model")
save_path = "out/my_prune/model"
# if hasattr(model, '_hf_hook'):
#     weights_map = get_weight_map(model)
            
# named_modules = dict(model.named_modules())
# for name, param in model.named_parameters():
#     if param.device.type == 'meta':
#         module_name, param_name = name.rsplit('.', 1)
#         module = named_modules[module_name]
#         delattr(module, param_name)
#         setattr(module, param_name, nn.Parameter(weights_map[name], False))
#     elif param.device.type != 'cpu':
#         module_name, param_name = name.rsplit('.', 1)
#         module = named_modules[module_name]
#         cpu_param = param.cpu()
#         delattr(module, param_name)
#         setattr(module, param_name, cpu_param)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
