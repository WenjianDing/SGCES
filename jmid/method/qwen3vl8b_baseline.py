# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model_path = '/root/autodl-tmp/Qwen3-VL-8B-Instruct'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
model = model.to('cuda')

test_data_path = "/root/autodl-tmp/ct_project/jmid_test_data.json"
example_data_path = "/root/autodl-tmp/ct_project/jmid_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/ct_project/jmid_qwen3vl8b_baseline.json"

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
    messages = [
        {
            "role": "user",
            "content": [
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
        generated_ids = model.generate(**inputs, max_new_tokens=128)
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
with open(save_data_path, 'w', encoding='utf-8') as outfile:
    json.dump(predict_data, outfile, ensure_ascii=False, indent=2)