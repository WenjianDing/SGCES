# coding=utf-8
import os
import json
import textwrap
import re
import torch
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

model_name = "openai/gpt-5.4"
#model_name = ""
client = OpenAI(
)

test_data_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"
save_data_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_gt_findings_structure_gpt54.json"


def get_result(full_prompt):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        max_tokens=1024
    )
    prediction = completion.choices[0].message.content
    # print('prediction')
    # print(prediction)
    return prediction


# 1. 结构化后处理函数 (保持不变)
def parse_and_align_json(raw_text):
    """
    清洗大模型输出的文本，提取JSON，并强制对齐必须的7个Key。
    如果模型漏了Key，填入None(null)；如果模型多加了Key，则丢弃。
    """
    required_keys = ["category", "anatomy", "observation", "status", "attributes"]
    fallback_result = {"findings": []}

    clean_text = raw_text.strip()
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1]
    if "```" in clean_text:
        clean_text = clean_text.split("```")[0]
    clean_text = clean_text.strip()

    try:
        parsed_data = json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"\n[Warning] JSON Decode Error. Raw text:\n{raw_text}\n")
        return fallback_result

    aligned_findings = []

    if isinstance(parsed_data, dict):
        findings_list = parsed_data.get("findings", parsed_data)
        if not isinstance(findings_list, list):
            findings_list = [findings_list]
    elif isinstance(parsed_data, list):
        findings_list = parsed_data
    else:
        findings_list = []

    for item in findings_list:
        if isinstance(item, dict):
            aligned_item = {key: item.get(key, None) for key in required_keys}
            aligned_findings.append(aligned_item)

    return {"findings": aligned_findings}


# ================= 开始处理流程 =================

with open(test_data_path, 'r', encoding='utf-8') as f1:
    test_data = json.load(f1)

test_data_length = len(test_data)

# 【核心修改 1：加载已有进度】
id2findings = {}
if os.path.exists(save_data_path):
    try:
        with open(save_data_path, 'r', encoding='utf-8') as f_save:
            id2findings = json.load(f_save)
        print(f"[*] 发现历史保存记录，已成功加载 {len(id2findings)} 条数据。将从剩余数据继续处理...")
    except json.JSONDecodeError:
        print("[!] 历史保存记录损坏或为空，将从头开始处理。")
        id2findings = {}
else:
    print("[*] 未发现历史保存记录，从头开始处理...")

for i in tqdm(range(test_data_length)):
    each_ct_data = test_data[i]
    uid = str(each_ct_data['uid'])

    # 【核心修改 2：检查当前 uid 是否已处理，处理过则跳过】
    if uid in id2findings:
        continue

    findings = each_ct_data['findings']

    # Prompt 保持原样
    prompt_raw = f"""
    #Role#
    You are a senior expert in structuring radiological data. Your task is to transform unstructured chest X-ray "Findings" text into a high-precision, structured JSON format to facilitate downstream automated logical reasoning.

    #Task#
    Carefully read the input "Findings" text, extract all clinical observations (including positive findings, explicitly stated negative findings, technical artifacts, and past history/medical devices), and map them into a standard JSON array. Do not omit any clinically valuable details, such as size, severity, and comparison with prior images.

    #Schema Definition#
    Strictly output a JSON object containing a `findings` list. Each item in the list must adhere to the following dictionary structure:
    {{
      "category": "Enum: [Lungs_and_Pleura, Cardiac_and_Mediastinum, Bones_and_Chest_Wall, Support_Devices, Technical_Quality, Other]",
      "anatomy": "Specific anatomical location (e.g., 'right apical', 'left lower lobe', 'thoracic spine', 'bilateral'). If not specified, set to null.",
      "observation": "Core observation or sign (e.g., 'pneumothorax', 'consolidation', 'cardiomegaly', 'clips', 'granuloma', 'clear').",
      "status": "Enum: [Present, Absent, Suspected]",
      "attributes": "Modifiers and detailed descriptions such as severity, borders, or shape (e.g., 'small to moderate', 'borderline', 'scattered', 'round'). If none, set to null."
    }}

    #Extraction Rules#
    1. Extract all negative expressions ("No", "without", "negative for", etc.) and mark their `status` as "Absent". For example: "No focal consolidation" -> observation: "consolidation", status: "Absent".
    2. Retain all degree adverbs and modifiers. For example: "Borderline cardiomegaly" -> observation: "cardiomegaly", attributes: "borderline", status: "Present".
    3. Extract normal expressions (e.g., "Lungs are clear", "Heart size normal") as separate, independent entries as well.
    4. Include any technical artifacts, positioning, or inspiration/volume issues (e.g., "mildly rotated", "low lung volumes"), and classify them under "Technical_Quality".
    5. You must return RAW JSON only. Do NOT wrap the output in Markdown code blocks (e.g., do not use ```json ... ```). Do not include any introductory, explanatory, or conversational text.

    #Input Findings:#
    {findings}
    """
    prompt = textwrap.dedent(prompt_raw).strip()

    # 调用大模型获取结果
    baseline_result = get_result(prompt)

    # 规范化输出格式
    structured_output = parse_and_align_json(baseline_result)
    # print('structured_output')
    # print(structured_output)
    # 将结果存入字典
    id2findings[uid] = structured_output

    # 【核心修改 3：实时保存】
    # 每跑完一条数据，就将其实时写入 JSON 文件
    # （大模型API调用通常要耗时几秒到十几秒，写一次几十KB的JSON只需几毫秒，不会影响整体性能）
    with open(save_data_path, 'w', encoding='utf-8') as outfile:
        json.dump(id2findings, outfile, ensure_ascii=False, indent=4)

print("\n[*] 全部数据处理完成！")