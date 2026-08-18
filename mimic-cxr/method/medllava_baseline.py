# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
model_path = '/root/autodl-tmp/llava-med-v1.5-mistral-7b-hf'
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # requires FA2
    device_map="auto"                          # multi-GPU ready
)

processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: "
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}<image>\n{% endif %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "\nASSISTANT:"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}"
    "{% endfor %}"
)
test_data_path = "/root/autodl-tmp/mimic_test_data.json"
example_data_path = "/root/autodl-tmp/mimic_ct_few_shot_examples.json"
save_data_path = "/root/autodl-tmp/mimic_llavamed_baseline.json"

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
    for j, ex in enumerate(each_examples):
        few_shot_str += f"Example {j + 1}:\nFindings: {ex['findings']}\nImpression: {ex['impression']}\n\n"

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
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]


    try:
        image = Image.open(Frontal_img_name).convert("RGB")
        prompt_filled = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            images=[image], text=prompt_filled, return_tensors="pt"
        ).to(model.device, torch.bfloat16)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256)
        predict_result = processor.decode(out[0], skip_special_tokens=True)
        if "ASSISTANT:" in predict_result:
            predict_result = predict_result.split("ASSISTANT:")[-1].strip()
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