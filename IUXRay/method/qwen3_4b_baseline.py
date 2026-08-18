# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = '/root/autodl-tmp/Qwen3-4B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
model = model.to('cuda')

test_data_path = "/root/autodl-tmp/ct_project/ct_data.json"
example_data_path = "/root/autodl-tmp/ct_project/ct_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/ct_project/qwen3_4b_baseline.json"

with open(test_data_path, 'r') as f1:
    test_data = json.load(f1)
with open(example_data_path, 'r') as f2:
    example_data = json.load(f2)
test_data_length = len(test_data)
predict_data = []
for i in tqdm(range(test_data_length)):
    each_predict = {}
    each_test_data = test_data[i]
    each_uid = each_test_data['uid']
    each_examples = example_data[each_uid]
    each_findings = each_test_data['findings']
    each_impression = each_test_data['impression']

    few_shot_str = ""
    for i, ex in enumerate(each_examples):
        few_shot_str += f"Example {i + 1}:\nFindings: {ex['findings']}\nImpression: {ex['impression']}\n\n"

    prompt_raw = f"""
**Role**
You are an expert radiologist.

**Task**
Synthesize the provided radiological findings into a concise, accurate diagnosis (Impression).

**Guidelines**
- Terminology: Adhere strictly to standard radiological terminology.
- Style: Closely follow the style, length, and logical deduction pattern of the examples.
- Output: Provide ONLY the text for the Impression. Do not include chatty text.

**Examples**
{few_shot_str.strip()}

---
**Current Case**
Findings: {each_findings}
Impression:
"""
    prompt = textwrap.dedent(prompt_raw).strip()
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        predict_result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # print('prompt')
        # print(prompt)
        # print('predict_result')
        # print(predict_result)

        each_predict['uid'] = each_uid
        each_predict['ground truth'] = each_impression
        each_predict['predicted'] = predict_result
        predict_data.append(each_predict)
    except:
        continue
with open(save_data_path, 'w') as outfile:
    json.dump(predict_data, outfile)