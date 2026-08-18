# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = '/root/autodl-tmp/Qwen2.5-VL-7B-Instruct'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen2.5-VL-7B-Instruct")
model = model.to('cuda')

test_data_path = "/root/autodl-tmp/mimic_test_data.json"
example_data_path = "/root/autodl-tmp/mimic_ct_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/mimic_qwen25vl_7b_baseline.json"

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

    Frontal_img_name = '/root/autodl-tmp/mimic_test_images/' + str(each_Frontal_name)


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
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predict_result = output_text[0]
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