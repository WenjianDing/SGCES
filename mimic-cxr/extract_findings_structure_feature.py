import json
import os
import torch
import numpy as np
import pickle
import ast
import re
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# 屏蔽警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_biomedclip_local(device):
    """
    复用之前的本地加载逻辑
    """
    model_name = "biomedclip_local"
    with open(".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX) and model_name not in _MODEL_CONFIGS):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)
    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained=".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin"
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def clean_and_parse_json(json_str, uid):
    """
    终极 JSON 抢救器：增加前置类型强转和安全的正则清理
    """
    if isinstance(json_str, dict):
        return json_str
    if isinstance(json_str, list):
        return {"findings": json_str}

    # 【改动核心】：防御极端的输入类型 (比如 float NaN 等)
    if not isinstance(json_str, str):
        if json_str is None or str(json_str).lower() == 'nan':
            return None
        # 强转为字符串以防万一
        json_str = str(json_str)

    # 1. 安全地替换非标准空格，并清理 Markdown 标记
    try:
        s = json_str.replace('\xa0', ' ').replace('&nbsp;', ' ')
        # 使用 \s* 吃掉可能存在的换行符和空格，忽略大小写
        s = re.sub(r'```json\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'```\s*', '', s)
        s = s.strip()
    except Exception as e:
        print(f"\n❌ [UID: {uid}] 字符串清理阶段直接报错: {e}")
        return None

    # 2. 先尝试用标准方法解析
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return {"findings": parsed}
        return parsed
    except json.JSONDecodeError:
        pass

    # 3. 暴力抢救模式：字符串括号匹配提取
    extracted_findings = []
    start_indices = [i for i, char in enumerate(s) if char == '{']

    for start in start_indices:
        stack = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                stack += 1
            elif s[i] == '}':
                stack -= 1
                if stack == 0:
                    dict_str = s[start:i + 1]
                    try:
                        parsed_dict = json.loads(dict_str)
                        if isinstance(parsed_dict, dict) and (
                                'category' in parsed_dict or 'observation' in parsed_dict):
                            if 'findings' not in parsed_dict:
                                extracted_findings.append(parsed_dict)
                    except:
                        try:
                            py_dict_str = dict_str.replace("null", "None").replace("true", "True").replace("false",
                                                                                                           "False")
                            parsed_dict = ast.literal_eval(py_dict_str)
                            if isinstance(parsed_dict, dict) and (
                                    'category' in parsed_dict or 'observation' in parsed_dict):
                                if 'findings' not in parsed_dict:
                                    extracted_findings.append(parsed_dict)
                        except:
                            pass
                    break

    if len(extracted_findings) > 0:
        print(f"⚠️ [UID: {uid}] 数据损坏或被截断，但成功抢救出 {len(extracted_findings)} 条完整结构！")
        return {"findings": extracted_findings}

    print(f"\n❌ [UID: {uid}] 彻底解析失败! 无法抢救任何信息。")
    return None

def serialize_finding_element(element):
    """
    强制对齐返回的 key，缺失就补 None
    """
    if not isinstance(element, dict):
        return None

    keys_to_extract = ["category", "anatomy", "observation", "status", "attributes"]
    extracted = {}

    for k in keys_to_extract:
        val = element.get(k, "None")
        if val is None or val == "":
            val = "None"
        elif isinstance(val, (dict, list)):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except:
                val = str(val)
        else:
            val = str(val).strip()

        extracted[k] = val
    serialized_dict = {}
    serialized = (f"[Category] {extracted['category']} "
                  f"[Anatomy] {extracted['anatomy']} "
                  f"[Observation] {extracted['observation']} "
                  f"[Status] {extracted['status']} "
                  f"[Attributes] {extracted['attributes']}")
    #print(serialized)
    serialized_dict['Category'] = extracted['category']
    serialized_dict['Anatomy'] = extracted['anatomy']
    serialized_dict['Observation'] = extracted['observation']
    serialized_dict['Status'] = extracted['status']
    serialized_dict['Attributes'] = extracted['attributes']
    #print(serialized_dict)
    return serialized, serialized_dict


def extract_structured_features(structured_dict, batch_size=128):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, tokenizer = load_biomedclip_local(device)

    uid_to_features = {}
    all_serialized_strings = []
    mapping_indices = []
    uid_to_serialized_dict = {}
    print("正在解析 JSON 并准备序列化文本...")
    for uid, json_str in tqdm(structured_dict.items(), desc="解析数据"):
        uid_to_features[uid] = []
        serialized_total_dict = []
        parsed_data = clean_and_parse_json(json_str, uid)
        if not parsed_data:
            continue

        findings_list = parsed_data.get("findings", [])

        # 兜底：如果没有 findings 列表，尝试将其视为单个结构的列表
        if not findings_list and isinstance(parsed_data, dict):
            if "category" in parsed_data or "observation" in parsed_data:
                findings_list = [parsed_data]

        if not findings_list or not isinstance(findings_list, list):
            continue

        # 抽取并打印信息
        for element in findings_list:
            serialized_str, serialized_dict = serialize_finding_element(element)
            if serialized_str:
                all_serialized_strings.append(serialized_str)
                serialized_total_dict.append(serialized_dict)
                mapping_indices.append(uid)
                print(f"✅ [UID: {uid}] 提取: {serialized_str}")
        uid_to_serialized_dict[uid] = serialized_total_dict
    print(f"\n解析完成。准备进入模型推理阶段，共获取到 {len(all_serialized_strings)} 条有效数据...")
    uid_to_serialized_dict_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_findings_structure_gpt54_dict.json"
    with open(uid_to_serialized_dict_path, 'w', encoding='utf-8') as outfile:
        json.dump(uid_to_serialized_dict, outfile, ensure_ascii=False, indent=4)

    all_embeddings = []
    context_length = 256

    if len(all_serialized_strings) > 0:
        with torch.no_grad():
            for i in tqdm(range(0, len(all_serialized_strings), batch_size), desc="模型推理"):
                batch_texts = all_serialized_strings[i: i + batch_size]

                tokens = tokenizer(batch_texts, context_length=context_length).to(device)
                features = model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)

                all_embeddings.append(features.cpu().numpy().astype(np.float32))

        full_matrix = np.vstack(all_embeddings)
        for idx, uid in enumerate(mapping_indices):
            uid_to_features[uid].append(full_matrix[idx])
    else:
        print("❌ 未提取到任何有效信息，特征提取已跳过。")

    return uid_to_features


if __name__ == "__main__":
    data_path = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_gt_findings_structure_gpt54.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    result_dict = extract_structured_features(raw_data)

    save_file = "uid_to_mimic_structured_features_gpt54.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"\n✅ 成功提取并保存至 {save_file}")