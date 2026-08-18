# coding=utf-8
import json
import os
import pickle
import ast
import re

import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


MODEL_NAME = r"...\bge-m3"
DATA_PATH = r"C:\Work\pricai\jmid\jmid_test\jmid_gt_findings_structure_qwen.json"
SERIALIZED_JSON_PATH = r"C:\Work\pricai\jmid\jmid_test\jmid_findings_structure_qwen_dict.json"
SAVE_FILE = "uid_to_jmid_structured_features_bgem3.pkl"

BATCH_SIZE = 12
MAX_LENGTH = 8192
USE_FP16 = True


def load_bgem3_model():
    """
    BGE-M3 official usage:
      model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
      model.encode(texts, batch_size=..., max_length=...)["dense_vecs"]

    We use only dense_vecs here because the downstream retrieval code expects
    one dense vector per serialized finding and computes similarity by np.dot.
    """
    return BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)


def clean_and_parse_json(json_str, uid):
    """
    Robustly parse model-generated structured JSON.
    """
    if isinstance(json_str, dict):
        return json_str
    if isinstance(json_str, list):
        return {"findings": json_str}

    if not isinstance(json_str, str):
        if json_str is None or str(json_str).lower() == "nan":
            return None
        json_str = str(json_str)

    try:
        s = json_str.replace("\xa0", " ").replace("&nbsp;", " ")
        s = re.sub(r"```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"```\s*", "", s)
        s = s.strip()
    except Exception as e:
        print(f"\n[UID: {uid}] String cleaning failed: {e}")
        return None

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return {"findings": parsed}
        return parsed
    except json.JSONDecodeError:
        pass

    extracted_findings = []
    start_indices = [i for i, char in enumerate(s) if char == "{"]

    for start in start_indices:
        stack = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                stack += 1
            elif s[i] == "}":
                stack -= 1
                if stack == 0:
                    dict_str = s[start:i + 1]
                    try:
                        parsed_dict = json.loads(dict_str)
                        if isinstance(parsed_dict, dict) and (
                            "category" in parsed_dict or "observation" in parsed_dict
                        ):
                            if "findings" not in parsed_dict:
                                extracted_findings.append(parsed_dict)
                    except Exception:
                        try:
                            py_dict_str = (
                                dict_str
                                .replace("null", "None")
                                .replace("true", "True")
                                .replace("false", "False")
                            )
                            parsed_dict = ast.literal_eval(py_dict_str)
                            if isinstance(parsed_dict, dict) and (
                                "category" in parsed_dict or "observation" in parsed_dict
                            ):
                                if "findings" not in parsed_dict:
                                    extracted_findings.append(parsed_dict)
                        except Exception:
                            pass
                    break

    if extracted_findings:
        print(f"[UID: {uid}] Recovered {len(extracted_findings)} structured findings.")
        return {"findings": extracted_findings}

    print(f"\n[UID: {uid}] JSON parsing failed.")
    return None


def serialize_finding_element(element):
    """
    Keep English keys, but values may be Japanese.
    Status should remain one of Present/Absent/Suspected for downstream logic.
    """
    if not isinstance(element, dict):
        return None, None

    keys_to_extract = ["category", "anatomy", "observation", "status", "attributes"]
    extracted = {}

    for key in keys_to_extract:
        val = element.get(key, "None")
        if val is None or val == "":
            val = "None"
        elif isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        else:
            val = str(val).strip()
        extracted[key] = val

    # Use English field tags for stable downstream parsing, while values can be Japanese.
    serialized = (
        f"[Category] {extracted['category']} "
        f"[Anatomy] {extracted['anatomy']} "
        f"[Observation] {extracted['observation']} "
        f"[Status] {extracted['status']} "
        f"[Attributes] {extracted['attributes']}"
    )

    serialized_dict = {
        "Category": extracted["category"],
        "Anatomy": extracted["anatomy"],
        "Observation": extracted["observation"],
        "Status": extracted["status"],
        "Attributes": extracted["attributes"],
    }
    return serialized, serialized_dict


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def extract_structured_features(structured_dict, batch_size=BATCH_SIZE):
    model = load_bgem3_model()

    uid_to_features = {}
    all_serialized_strings = []
    mapping_indices = []
    uid_to_serialized_dict = {}

    print("Parsing structured JSON and preparing Japanese serialized texts...")
    for uid, json_str in tqdm(structured_dict.items(), desc="Parsing data"):
        uid_to_features[uid] = []
        serialized_total_dict = []
        parsed_data = clean_and_parse_json(json_str, uid)
        if not parsed_data:
            continue

        findings_list = parsed_data.get("findings", [])

        if not findings_list and isinstance(parsed_data, dict):
            if "category" in parsed_data or "observation" in parsed_data:
                findings_list = [parsed_data]

        if not findings_list or not isinstance(findings_list, list):
            continue

        for element in findings_list:
            serialized_str, serialized_dict = serialize_finding_element(element)
            if serialized_str:
                all_serialized_strings.append(serialized_str)
                serialized_total_dict.append(serialized_dict)
                mapping_indices.append(uid)
                print(f"[UID: {uid}] Extracted: {serialized_str}")

        uid_to_serialized_dict[uid] = serialized_total_dict

    print(f"\nParsed {len(all_serialized_strings)} valid serialized findings.")
    with open(SERIALIZED_JSON_PATH, "w", encoding="utf-8") as outfile:
        json.dump(uid_to_serialized_dict, outfile, ensure_ascii=False, indent=4)

    if not all_serialized_strings:
        print("No valid findings were extracted. Skip embedding.")
        return uid_to_features

    all_embeddings = []
    for i in tqdm(range(0, len(all_serialized_strings), batch_size), desc="Encoding with BGE-M3"):
        batch_texts = all_serialized_strings[i:i + batch_size]

        output = model.encode(
            batch_texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
        )
        dense_vecs = output["dense_vecs"]

        # Keep cosine-by-dot behavior compatible with existing retrieval code.
        dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
        dense_vecs = l2_normalize(dense_vecs).astype(np.float32)
        all_embeddings.append(dense_vecs)

    full_matrix = np.vstack(all_embeddings)
    for idx, uid in enumerate(mapping_indices):
        uid_to_features[uid].append(full_matrix[idx])

    return uid_to_features


if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    result_dict = extract_structured_features(raw_data)

    with open(SAVE_FILE, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"\nSaved structured BGE-M3 features to {SAVE_FILE}")
