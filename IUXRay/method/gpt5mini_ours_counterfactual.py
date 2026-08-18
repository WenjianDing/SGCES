# coding=utf-8
import os
import json
import textwrap
import torch
import base64
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# 确保在代码开头初始化你的 OpenAI 客户端 (需补充 api_key 等信息)
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

model_name = "openai/gpt-5-mini"

test_data_path = "C:/Work/pricai/ct_data.json"
# 这里的 example_data_path 是你之前保存的相似度映射字典 (Top-3 IDs)
example_mapping_path = "C:/Work/pricai/ours/final_sim_mapping_top3_symbolic_442.json"
# 新增：反事实编辑后的结构化特征字典路径
example_edited_struct_path = "C:/Work/pricai/ours/final_dual_counterfactual_edited_top3_442.json"

save_data_path = "C:/Work/pricai/result/ours/gpt5mini_rag_symbolic_442_dual_counterfactual_results.json"

image_dir = 'C:/Work/pricai/dataset/ct_images/images_normalized/'

os.makedirs(os.path.dirname(save_data_path), exist_ok=True)


# --- 2. 预处理数据：构建 UID 索引词典 ---
print("正在预处理数据并构建索引...")
with open(test_data_path, 'r', encoding='utf-8') as f1:
    test_data = json.load(f1)

with open(example_mapping_path, 'r', encoding='utf-8') as f2:
    sim_mapping = json.load(f2)

# 新增：加载反事实修剪后的结构化要素数据
with open(example_edited_struct_path, 'r', encoding='utf-8') as f3:
    edited_mapping = json.load(f3)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 将原始数据转化为 {uid: {findings, impression}} 格式，方便瞬间查询
uid_to_content = {
    str(item['uid']): {
        'findings': item['findings'],
        'impression': item['impression']
    } for item in test_data
}

test_data_length = len(test_data)
predict_data = []
processed_uids = set()

if os.path.exists(save_data_path):
    try:
        with open(save_data_path, 'r', encoding='utf-8') as f:
            predict_data = json.load(f)
        # 记录所有已经成功处理过的 uid
        processed_uids = {item['uid'] for item in predict_data}
        print(f"发现已有进度！已加载 {len(processed_uids)} 条预测结果，将从断点处继续...")
    except json.JSONDecodeError:
        print("警告：已存在的保存文件格式损坏，将从头开始覆盖。")

# --- 3. 循环生成 ---
for i in tqdm(range(test_data_length)):
    each_predict = {}
    each_test_data = test_data[i]
    each_uid = str(each_test_data['uid'])
    if each_uid in processed_uids:
        continue
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

    try:
        # 处理输入
        #image = Image.open(Frontal_img_path)
        base64_image = encode_image(Frontal_img_path)

        # 准备多模态输入
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        predict_result = completion.choices[0].message.content
        # print('prompt')
        # print(prompt)
        # print('predict_result')
        # print(predict_result)

        # 整理输出
        each_predict['uid'] = each_uid
        each_predict['ground truth'] = each_impression
        each_predict['predicted'] = predict_result.strip()
        predict_data.append(each_predict)

        # 【新增】每次预测成功后，立刻覆盖保存 JSON
        with open(save_data_path, 'w', encoding='utf-8') as outfile:
            json.dump(predict_data, outfile, ensure_ascii=False, indent=4)

        # 记录到 set 中，防止后续逻辑错乱
        processed_uids.add(each_uid)

    except Exception as e:
        #print(f"\n[Error] UID {each_uid} 处理失败，跳过。错误信息: {str(e)}")
        continue

