# coding=utf-8
import os
import json
import textwrap
import requests
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from tqdm import tqdm
from transformers import pipeline, set_seed
#we use seed = 42/43/44
seed = 44
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.5,
    "top_p": 0.95,
}
model_path = '/root/autodl-tmp/medgemma-4b-it'
pipe = pipeline(
    "image-text-to-text",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device="cuda",
)


test_data_path = "/root/autodl-tmp/ct_project/jmid_test_data.json"
example_data_path = "/root/autodl-tmp/ct_project/jmid_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/ct_project/jmid_medgemma_baseline_r3.json"

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
    #each_Frontal_name = each_test_data['Frontal']
    #each_Lateral_name = each_test_data['Lateral']

    #Frontal_img_name = '/root/autodl-tmp/ct_project/mimic_test_images/' + str(each_Frontal_name)
    #Lateral_img_name = '/root/autodl-tmp/ct_images/images_normalized/' + str(each_Frontal_name)

    few_shot_str = ""
    for j, ex in enumerate(each_examples):
        few_shot_str += f"Example {j + 1}:\nFindings: {ex['findings']}\nImpression: {ex['impression']}\n\n"

    prompt_raw = f"""
**Role**
You are an expert radiologist.

**Task**
The provided Findings are written in Japanese. Synthesize them into a concise, accurate radiological Impression written in Japanese.

**Guidelines**
- Language: Output the Impression in Japanese. Do not translate it into English.
- Terminology: Use standard Japanese radiological terminology. Common medical abbreviations such as MRI, DWI, CRM, EMVI, MRF, T1, and T2 may be kept in English when appropriate.
- Faithfulness: Base the Impression only on the provided Findings. Do not add unsupported diagnoses.
- Style: Closely follow the style, length, and logical deduction pattern of the examples.
- Output: Provide ONLY the Impression text. Do not include explanations, labels, markdown, or chatty text.

**Examples**
{few_shot_str.strip()}

---
**Current Case**
Findings Japanese: {each_findings}
Impression Japanese:
"""
    prompt = textwrap.dedent(prompt_raw).strip()
    #image = Image.open(requests.get(Frontal_img_name, headers={"User-Agent": "example"}, stream=True).raw)


    try:
        #image = Image.open(Frontal_img_name)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        output = pipe(
            text=messages,
            generate_kwargs=generation_kwargs,
        )
        predict_result = output[0]["generated_text"][-1]["content"]
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
with open(save_data_path, 'w', encoding='utf-8') as outfile:
    json.dump(predict_data, outfile, ensure_ascii=False, indent=2)