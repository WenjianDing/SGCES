import os
import json
import re
import base64
import time

from tqdm import tqdm
from openai import OpenAI
import textwrap
import numpy as np


# 初始化 OpenRouter 客户端
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

# 定义用于评估的三个多模态大模型
# **这里我们最终只使用了gemini3.5的结果**
EVAL_MODELS = {
    "model1": "openai/gpt-5.4",
    "model2": "anthropic/claude-sonnet-4.6",
    "model3": "google/gemini-3.5-flash"
}


def calculate_evaluation_statistics(result_path):
    # 1. 读取评估结果 JSON
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"找不到文件: {result_path}")
        return

    valid_results = []

    # 2. 数据清洗：剔除任何包含 [0, 0, 0] 的 uid
    for item in data:
        scores = item.get("eval_scores", {})
        is_valid = True

        # 检查该 uid 下所有模型的打分
        for model_name, score_list in scores.items():
            if score_list == [0, 0, 0]:
                is_valid = False
                break  # 只要有一个模型失败，整个 uid 作废

        if is_valid:
            valid_results.append(item)

    # 打印清洗概况
    total_samples = len(data)
    valid_samples = len(valid_results)
    print(f"数据清洗报告:")
    print(f"总样本数: {total_samples}")
    print(f"有效样本数: {valid_samples}")
    print(f"因包含 [0, 0, 0] 被剔除的样本数: {total_samples - valid_samples}")

    if not valid_results:
        print("警告: 清洗后没有剩余的有效数据，无法计算统计信息！")
        return

    # 3. 初始化分数累加器
    # 指标顺序对应: [Accuracy, Comprehensiveness, Faithfulness]
    stats = {
        "model1 (GPT)": {"Acc": [], "Comp": [], "Faith": []},
        "model2 (Claude)": {"Acc": [], "Comp": [], "Faith": []},
        "model3 (Gemini)": {"Acc": [], "Comp": [], "Faith": []}
    }

    # 映射字典，方便对应 JSON 里的 key 到我们展示的名称
    model_key_mapping = {
        "model1": "model1 (GPT)",
        "model2": "model2 (Claude)",
        "model3": "model3 (Gemini)"
    }

    # 4. 收集有效分数
    for item in valid_results:
        scores = item["eval_scores"]
        for json_key, display_name in model_key_mapping.items():
            if json_key in scores:
                stats[display_name]["Acc"].append(scores[json_key][0])
                stats[display_name]["Comp"].append(scores[json_key][1])
                stats[display_name]["Faith"].append(scores[json_key][2])

    # 5. 计算均值并格式化输出 (保留1位小数，方便直接填入 LaTeX 表格)
    print("最终平均分 (基于有效样本):")
    print(f"{'评委模型':<20} | {'Acc':<6} | {'Comp':<6} | {'Faith':<6}")
    print("-" * 45)

    for model, metrics in stats.items():
        # 计算均值，如果列表为空则返回0
        avg_acc = np.mean(metrics["Acc"]) if metrics["Acc"] else 0
        avg_comp = np.mean(metrics["Comp"]) if metrics["Comp"] else 0
        avg_faith = np.mean(metrics["Faith"]) if metrics["Faith"] else 0

        print(f"{model:<20} | {avg_acc:<6.2f} | {avg_comp:<6.2f} | {avg_faith:<6.2f}")



def encode_image_to_base64(image_path):
    """将本地图片读取并转换为 Base64 编码"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def get_llm_scores(model_id, prediction, reference, findings, frontal_img_path):
    """
    调用多模态 LLM 进行评估，并返回 [Acc, Comp, Faith] 分数列表
    """
    # 构造文本部分的 Prompt
    text_prompt = f"""
    You are an extremely critical and skeptical senior radiologist auditing AI-generated medical reports. Your task is to evaluate a "Predicted Impression" against a human-written "Ground Truth Impression", based on the provided [Image] and [Findings].

### Inputs:
- [Image]: (The provided CT scan image)
- [Findings]: {findings}
- [Ground Truth Impression]: {reference}
- [Predicted Impression]: {prediction}

### CRITICAL SCORING RULES - AVOID SCORE INFLATION:
You are currently suffering from lenience bias. You MUST correct this. 
1. The DEFAULT score for any category is 3 (Average/Acceptable). 
2. You must actively look for reasons to DEDUCT points. 
3. A score of 5 is virtually impossible. It means the predicted text is indistinguishable from the senior radiologist's ground truth in every conceivable way. If there is even one extra word, one missing benign detail, or slightly suboptimal phrasing, the score CANNOT be a 5.

### Evaluation Criteria:

#### 1. Clinical Accuracy & Factual Correctness
* 5: Literally flawless. Zero hallucinations, zero false positives, perfect anatomical localization identical to Ground Truth.
* 4: Clinically safe, but uses slightly less precise terminology than the Ground Truth.
* 3: (DEFAULT) Captures the main idea, but contains a minor hallucination, assumes implicitly unstated facts, or slightly distorts the Ground Truth.
* 2: Misses a significant secondary diagnosis or introduces a potentially harmful false positive.
* 1: Completely incorrect, misses the primary diagnosis, or presents dangerous clinical misinformation.

#### 2. Completeness & Information Coverage
* 5: Matches the Ground Truth coverage perfectly. Not a single detail, critical negative, or location is missed.
* 4: Captures all primary diagnoses, but misses exactly ONE trivial/benign observation mentioned in the Ground Truth.
* 3: (DEFAULT) Misses more than one benign detail OR misses a secondary but clinically relevant context (e.g., forgets to mention the specific lobe).
* 2: Misses a primary, actionable diagnosis present in the Ground Truth.
* 1: Fails to capture the core diagnoses entirely. Unacceptably brief.

#### 3. Conciseness, Synthesis & Professionalism
* 5: Maximum information density. Perfect synthesis of findings without any unnecessary repetition.
* 4: Good synthesis, but contains 1-2 unnecessary sentences or slightly wordy phrasing compared to Ground Truth.
* 3: (DEFAULT) Noticeable copy-pasting from the "Findings" section. Fails to properly summarize or prioritize the most severe findings first.
* 2: Rambling, poorly structured, or buries the critical diagnosis at the bottom.
* 1: Incomprehensible, not formatted as an impression, or highly unprofessional.

### Output Format:
You must return your evaluation STRICTLY as a list of three integers representing the scores for [Accuracy, Completeness, Conciseness] in that exact order. 
Do NOT output any reasoning, explanations, markdown formatting, or additional text.

Example output:
[3, 4, 2]
    """
    prompt = textwrap.dedent(text_prompt).strip()
    if True:
        #print(frontal_img_path)
        base64_image = encode_image_to_base64(frontal_img_path)
        # 准备多模态输入
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = client.chat.completions.create(
            model=model_id,
            messages=messages
        )
        result_text = completion.choices[0].message.content
        # print('prompt')
        # print(prompt)
        # print('result_text')
        # print(result_text)

        match = re.search(r'\[.*?\]', result_text, re.DOTALL)
        if match:
            scores = json.loads(match.group(0))
            if isinstance(scores, list) and len(scores) == 3:
                return [int(s) for s in scores]

        #print(f"\nWarning: Unexpected format from {model_id}: {result_text}")
        return [0, 0, 0]

    # except Exception as e:
    #     #print(f"\nError calling API for {model_id}: {str(e)}")
    #     return [0, 0, 0]


def llm_evaluation(data_path, ct_data_path, image_base_dir, save_path):
    # 1. 读取基础评估数据
    with open(data_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    # 2. 读取并构建 CT 数据的映射字典 (UID -> Context)
    with open(ct_data_path, 'r', encoding='utf-8') as f:
        ct_data_list = json.load(f)

    # 将 ct_data 转换为以 uid 为 key 的字典，方便 O(1) 查找
    # 注意：为了防呆，这里统一将 uid 强转为字符串形式
    ct_mapping = {str(item["uid"]): item for item in ct_data_list}

    predict_data = []
    processed_uids = set()
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                predict_data = json.load(f)
            # 记录所有已经成功处理过的 uid
            processed_uids = {item['uid'] for item in predict_data}
            print(f"发现已有进度！已加载 {len(processed_uids)} 条预测结果，将从断点处继续...")
        except json.JSONDecodeError:
            print("警告：已存在的保存文件格式损坏，将从头开始覆盖。")


    print(f"Starting multimodal evaluation for {len(baseline_data)} samples...")
    for i in tqdm(range(len(baseline_data))):
        if i % 20 == 0:
            each_result = baseline_data[i]
            uid = str(each_result.get('uid', str(i)))
            if uid in processed_uids:
                continue
            try:
                predictions = each_result["predicted"]
                references = each_result["ground truth"]
            except:
                predictions = each_result.get("best_impression", "")
                references = each_result.get("ground_truth", "")

            # 3. 从映射字典中提取 Context 信息
            context_data = ct_mapping.get(uid, {})
            findings = context_data.get("findings", "No findings available.")

            # 拼接图片的绝对路径
            frontal_filename = context_data.get("Frontal", "")

            frontal_path = os.path.join(image_base_dir, frontal_filename) if frontal_filename else None

            current_eval = {
                "uid": uid,
                "prediction": predictions,
                "ground_truth": references,
                "eval_scores": {}
            }
            #print('current_eval')
            #print(current_eval)
            # 4. 遍历调用三个模型进行多模态评估
            for model_name, model_id in EVAL_MODELS.items():
                scores = get_llm_scores(
                    model_id=model_id,
                    prediction=predictions,
                    reference=references,
                    findings=findings,
                    frontal_img_path=frontal_path
                )
                current_eval["eval_scores"][model_name] = scores

            predict_data.append(current_eval)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(predict_data, f, ensure_ascii=False, indent=4)
            processed_uids.add(uid)

if __name__ == '__main__':

    data_path = r"C:\Work\pricai\result\ours\medgemma_rag_symbolic_226_dual_counterfactual_results.json"
    # 你存放 findings 和文件名映射的 ct_data json 路径
    ct_data_path = r"C:\Work\pricai\ct_data.json"
    # 【新增】你的图片实际存放在哪里的本地绝对路径
    image_base_dir = r"C:\Work\pricai\dataset\ct_images\images_normalized"
    # 结果保存路径
    save_path = r"C:\Work\pricai\result\llm_as_a_judge\medgemma_final.json"

    llm_evaluation(data_path, ct_data_path, image_base_dir, save_path)
    time.sleep(5)
    calculate_evaluation_statistics(save_path)