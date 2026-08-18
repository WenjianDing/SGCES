# coding=utf-8
import os
import json
import textwrap
import base64
from tqdm import tqdm
from openai import OpenAI

# 确保在代码开头初始化你的 OpenAI 客户端 (需补充 api_key 等信息)
client = OpenAI(
)

model_name = "openai/gpt-3.5-turbo"

test_data_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"
example_data_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_ct_few_shot_examples.json"
save_data_path = r"C:\Work\pricai\result\mimic\mimic_gpt35_baseline.json"

# 为了防止保存路径的文件夹不存在导致报错，提前创建目录
os.makedirs(os.path.dirname(save_data_path), exist_ok=True)

with open(test_data_path, 'r', encoding='utf-8') as f1:
    test_data = json.load(f1)
with open(example_data_path, 'r', encoding='utf-8') as f2:
    example_data = json.load(f2)

# ==========================================
# 新增：断点续传逻辑，读取已存在的进度
# ==========================================
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


# ==========================================
# 辅助函数：将本地图片转为 base64 (OpenAI Vision API 标准要求)
# ==========================================


test_data_length = len(test_data)

for i in tqdm(range(test_data_length)):
    each_predict = {}
    each_test_data = test_data[i]
    each_uid = each_test_data['uid']

    # 【新增】如果当前 uid 已经处理过，直接跳过
    if each_uid in processed_uids:
        continue

    each_examples = example_data[each_uid]
    each_findings = each_test_data['findings']
    each_impression = each_test_data['impression']


    few_shot_str = ""
    for j, ex in enumerate(each_examples):  # 修正了变量覆盖引起的潜在bug (原来也是i)
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
    {few_shot_str.strip() if few_shot_str else "No examples available."}

    ---
    **Current Case**
    Findings: {each_findings}
    Impression:
    """
    prompt = textwrap.dedent(prompt_raw).strip()

    try:
        # 转换本地图片为 base64

        # 修正图片传入格式：OpenAI API 要求 image_url 和 base64 协议
        messages = [
            {
                "role": "user",
                "content": prompt
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

        each_predict['uid'] = each_uid
        each_predict['ground truth'] = each_impression
        each_predict['predicted'] = predict_result

        # 将新结果加入列表
        predict_data.append(each_predict)

        # 【新增】每次预测成功后，立刻覆盖保存 JSON
        with open(save_data_path, 'w', encoding='utf-8') as outfile:
            json.dump(predict_data, outfile, ensure_ascii=False, indent=4)

        # 记录到 set 中，防止后续逻辑错乱
        processed_uids.add(each_uid)

    except Exception as e:
        # 【修改】打印出具体的错误原因，避免 API 报错（如限流、图片不存在）时你不知道发生了什么
        print(f"\n[Error] UID {each_uid} 处理失败，跳过。错误信息: {str(e)}")
        continue