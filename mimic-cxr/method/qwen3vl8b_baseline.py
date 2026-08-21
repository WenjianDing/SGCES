# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed
#we use seed = 42/43/44
seed = 44
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)


model_path = '/root/autodl-tmp/Qwen3-VL-8B-Instruct'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
model = model.to('cuda')

test_data_path = "/root/autodl-tmp/ct_project/mimic_test_data.json"
example_data_path = "/root/autodl-tmp/ct_project/mimic_ct_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/ct_project/mimic_qwen3vl8b_baseline_r3.json"

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
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Frontal_img_name,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


    try:

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.5, top_p=0.95)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        predict_result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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