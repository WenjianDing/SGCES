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
# --- 配置路径 ---
model_path = '/root/autodl-tmp/Qwen3-VL-8B-Instruct'
test_data_path = "/root/autodl-tmp/ct_project/mimic_test_data.json"
# 这里的 example_data_path 是你之前保存的相似度映射字典 (Top-3 IDs)
example_mapping_path = "/root/autodl-tmp/ct_project/mimic_sim_mapping_top3_symbolic_442.json"
# 新增：反事实编辑后的结构化特征字典路径
example_edited_struct_path = "/root/autodl-tmp/ct_project/mimic_dual_counterfactual_top3_442.json"

save_data_path = "/root/autodl-tmp/ct_project/mimic_qwen3vl8b_ours_r3.json"
image_dir = '/root/autodl-tmp/ct_project/mimic_test_images/'
# with open("/root/autodl-tmp/ct_project/gt_findings_structure_gpt54_dict.json", "r") as f3:
#     uid_to_struct = json.load(f3)
# --- 1. 加载模型与处理器 ---
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

# 新增：加载反事实修剪后的结构化要素数据
with open(example_edited_struct_path, 'r', encoding='utf-8') as f3:
    edited_mapping = json.load(f3)

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

    # 获取检索到的相似 ID 列表和对应的结构化要素列表
    similar_uids = sim_mapping.get(each_uid, [])
    edited_structs = edited_mapping.get(each_uid, [])

    each_findings = each_test_data['findings']
    each_impression = each_test_data['impression']
    each_Frontal_name = each_test_data['Frontal']

    Frontal_img_path = os.path.join(image_dir, str(each_Frontal_name))

    # --- 构造 Few-shot 字符串 (方式二：自然语言文本 + 对齐后的结构化引导) ---
    few_shot_str = ""
    for idx, sim_id in enumerate(similar_uids):
        sim_id = str(sim_id)
        if sim_id in uid_to_content:
            content = uid_to_content[sim_id]
            # 获取当前参考样本对应的已对齐结构化要素
            current_struct = edited_structs[idx] if idx < len(edited_structs) else []
            # 将其紧凑地序列化为 JSON 文本块
            struct_json_str = json.dumps(current_struct, ensure_ascii=False, indent=2)

            few_shot_str += f"Example {idx + 1}:\n"
            few_shot_str += f"Original Findings: {content['findings']}\n"
            few_shot_str += f"Aligned Core Entities (JSON): {struct_json_str}\n"
            few_shot_str += f"Impression: {content['impression']}\n\n"

    # --- 构造 Prompt ---
    # 在 Guidelines 中显式指导模型：利用 Aligned Core Entities 作为映射桥梁
    prompt_raw = f"""
**Role**
You are an expert radiologist.

**Task**
Synthesize the provided radiological findings into a concise, accurate diagnosis (Impression).

**Guidelines**
- Terminology: Adhere strictly to standard radiological terminology.
- Style: Closely follow the style, length, and logical deduction pattern of the examples.
- **Structural Alignment**: Each historical example contains an "Aligned Core Entities (JSON)" block, which represents the pure clinical elements from that case that genuinely map onto the current patient's situation. Use this block as a reasoning anchor to learn how valid core findings bridge to the final Impression, bypassing any unaligned noise in the original reference text.
- Output: Provide ONLY the text for the Impression. Do not include chatty text.

**Examples**
{few_shot_str.strip() if few_shot_str else "No examples available."}

---
**Current Case**
Findings: {each_findings}
Impression:
"""
    prompt = textwrap.dedent(prompt_raw).strip()

    # 准备多模态输入
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Frontal_img_path,
                },
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
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.5, top_p=0.95)

        # 裁剪掉 Prompt 部分
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