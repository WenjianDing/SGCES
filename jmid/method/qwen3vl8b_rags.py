# coding=utf-8
import os
import json
import textwrap
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- 配置路径 ---
model_path = '/root/autodl-tmp/Qwen3-VL-8B-Instruct'
test_data_path = "/root/autodl-tmp/ct_project/jmid_test_data.json"
# 这里的 example_data_path 是你之前保存的相似度映射字典 (Top-3 IDs)
example_mapping_path = "/root/autodl-tmp/ct_project/jmid_medrag_top3.json"
save_data_path = "/root/autodl-tmp/ct_project/jmid_medrag_results.json"
#image_dir = '/root/autodl-tmp/ct_project/mimic_test_images/'

# --- 1. 加载模型与处理器 --
print("正在加载 Qwen3-VL 模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# --- 2. 预处理数据：构建 UID 索引词典 ---
print("正在预处理数据并构建索引...")
with open(test_data_path, 'r', encoding='utf-8') as f1:
    test_data = json.load(f1)

with open(example_mapping_path, 'r', encoding='utf-8') as f2:
    sim_mapping = json.load(f2)

# 将原始数据转化为 {uid: {findings, impression}} 格式，方便瞬间查询
uid_to_content = {
    str(item['uid']): {
        'findings': item['findings'],
        'impression': item['impression']
    } for item in test_data
}

test_data_length = len(test_data)
predict_data = []

# --- 3. 循环生成 ---
for i in tqdm(range(test_data_length)):
    each_predict = {}
    each_test_data = test_data[i]
    each_uid = str(each_test_data['uid'])

    # 获取检索到的相似 ID 列表
    similar_uids = sim_mapping.get(each_uid, [])
    #import random
    #similar_uids = [str(x) for x in random.sample(range(1, 3996), 3)]
    each_findings = each_test_data['findings']
    each_impression = each_test_data['impression']
    #each_Frontal_name = each_test_data['Frontal']

    #Frontal_img_path = os.path.join(image_dir, str(each_Frontal_name))

    # --- 构造 Few-shot 字符串 ---
    few_shot_str = ""
    example_count = 0
    for sim_id in similar_uids:
        sim_id = str(sim_id)
        if sim_id in uid_to_content:
            content = uid_to_content[sim_id]
            example_count += 1
            few_shot_str += f"Example {example_count}:\nFindings: {content['findings']}\nImpression: {content['impression']}\n\n"

    # --- 构造 Prompt ---
    # 强调逻辑推理模式和医学术语一致性
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

    # 准备多模态输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # 生成结果：max_new_tokens 设置为 128 足够 Impression 使用
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        # 裁剪掉 Prompt 部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        predict_result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print('prompt')
        # print(prompt)
        # print("predict_result")
        # print(predict_result)
        # 整理输出
        each_predict['uid'] = each_uid
        each_predict['ground truth'] = each_impression
        each_predict['predicted'] = predict_result.strip()
        predict_data.append(each_predict)

    except Exception as e:
        print(f"\nError processing UID {each_uid}: {e}")
        continue

# --- 4. 保存结果 ---
with open(save_data_path, 'w', encoding='utf-8') as outfile:
    json.dump(predict_data, outfile, ensure_ascii=False, indent=4)

print(f"\n✅ 推理完成！预测结果已保存至: {save_data_path}")