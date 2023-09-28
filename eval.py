from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorWithPadding
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_from_disk
from tqdm import tqdm

from lib.data import get_loaders


parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, default="~/models/models--Phind--Phind-CodeLlama-34B-v2")
parser.add_argument('--data_path', type=str, default="~/datasets/nickrosh/Evol-Instruct-Code-80k-v1")
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()
print(args)


def get_attention_mask(attention_mask_bool, device=None):
    if device is None:
        device = attention_mask_bool.device
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

model_dir = Path(args.model_dir).expanduser()
data_path = Path(args.data_path).expanduser()
batch_size = args.batch_size

config = AutoConfig.from_pretrained(model_dir)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = {}
device_map['model.embed_tokens'] = 0
for i in range(len(model.model.layers)):
    device_map[f'model.layers.{i}'] = 'cpu'
device_map['model.norm'] = 'cpu'
device_map['lm_head'] = 'cpu'

model = load_checkpoint_and_dispatch(
    model,
    checkpoint= model_dir,
    device_map=device_map,
    dtype=torch.float16,
    no_split_module_classes=['LlamaDecoderLayer']
)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

collater = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

testdataset = load_from_disk(data_path)
testdataset = testdataset['train']
testdataset = testdataset.shard(num_shards=100, index=0)
prompt_template = "### System Prompt\nYou are an intelligent programming assistant.\n\n### User Message\n{instruction}\n\n### Assistant\n"
# prompt_template = "Below is an instruction that describes a request regarding programming. "\
#                   "Write a response that appropriately completes the request.\n\n"\
#                   "### Instruction:\n{instruction}\n\n### Response:\n"

def preprocess(sample, tokenizer=tokenizer):
    prompt = prompt_template.format(instruction=sample['instruction'])
    response = sample['output']
    tokenized_result = tokenizer([prompt, response], return_attention_mask=False)
    prompt = tokenized_result['input_ids'][0]
    response = tokenized_result['input_ids'][1][1:]
    input_ids = prompt + response
    sequence_length = len(input_ids)
    prompt_length = len(prompt)
    return {'input_ids': input_ids,
            'sequence_length': sequence_length,
            'prompt_length': prompt_length}

testdataset = testdataset.map(preprocess, remove_columns=['instruction', 'output'], desc='Preprocessing data', num_proc=8)
testdataset.set_format(type='torch', columns=['input_ids'], output_all_columns=True)
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, collate_fn=collater)

nlls = []
skipped = 0
with tqdm(total=len(testdataset), unit='sample') as pbar:
    for batch in testloader:
        batch_size = batch['input_ids'].shape[0]
        try:
            input_ids = batch['input_ids']
            labels = input_ids.clone()
            sequence_length = batch['sequence_length']
            prompt_length = batch['prompt_length']
            for i in range(batch_size):
                labels[i, :prompt_length[i]] = -100
                labels[i, sequence_length[i]+1:] = -100
                
            # attention_mask_bool = batch['attention_mask']
            # attention_mask = get_attention_mask(attention_mask_bool)
            
            outputs = model(input_ids=input_ids, attention_mask=batch['attention_mask'], labels=labels)
            nlls.append(outputs.loss.item())
        except torch.cuda.OutOfMemoryError:
            skipped += batch_size

        pbar.update(batch_size)
        pbar.set_postfix({'OOM_skiped': skipped, 'nll': nlls[-1] if len(nlls) > 0 else None})
        torch.cuda.empty_cache()

nlls = torch.tensor(nlls).mean()
ppl = torch.exp(nlls)
print(f'PPL: {ppl}')