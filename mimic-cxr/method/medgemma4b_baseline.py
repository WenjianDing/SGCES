# coding=utf-8
import os
import json
import textwrap
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
model_path = '/root/autodl-tmp/medgemma-4b-it'
pipe = pipeline(
    "image-text-to-text",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device="cuda",
)


test_data_path = "/root/autodl-tmp/ct_project/mimic_test_data.json"
example_data_path = "/root/autodl-tmp/ct_project/mimic_ct_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/ct_project/mimic_medgemma_baseline.json"

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
    each_Frontal_name = each_test_data['Frontal']
    #each_Lateral_name = each_test_data['Lateral']

    Frontal_img_name = '/root/autodl-tmp/ct_project/mimic_test_images/' + str(each_Frontal_name)
    #Lateral_img_name = '/root/autodl-tmp/ct_images/images_normalized/' + str(each_Frontal_name)

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
    #image = Image.open(requests.get(Frontal_img_name, headers={"User-Agent": "example"}, stream=True).raw)


    try:
        image = Image.open(Frontal_img_name)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        output = pipe(text=messages, max_new_tokens=128, max_length=None)
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
with open(save_data_path, 'w') as outfile:
    json.dump(predict_data, outfile)