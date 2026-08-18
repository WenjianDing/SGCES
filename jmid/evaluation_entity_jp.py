# coding=utf-8
import json
import os
import re
import csv
import random
import hashlib
import unicodedata
from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except Exception:
    np = None
    linear_sum_assignment = None


# ==================== 配置区 ====================

MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"

DATA_PATHS = [
    "/root/autodl-tmp/jmid_internvl3_8b_baseline.json",
    "/root/autodl-tmp/jmid_medcpt_results.json",
"/root/autodl-tmp/jmid_medgemma_baseline.json",
"/root/autodl-tmp/jmid_medgemma_ours_results.json",
"/root/autodl-tmp/jmid_medrag_results.json",
"/root/autodl-tmp/jmid_qwen2vl_7b_baseline.json",
"/root/autodl-tmp/jmid_qwen3_4b_baseline.json",
"/root/autodl-tmp/jmid_qwen3_8b_baseline.json",
"/root/autodl-tmp/jmid_qwen25_7b_baseline.json",
"/root/autodl-tmp/jmid_qwen25vl_7b_baseline.json",
"/root/autodl-tmp/jmid_radgraph_results.json",
]


# 用第一个结果文件作为采样来源，10个文件都评估同一批uid
SAMPLE_SOURCE_PATH = DATA_PATHS[0]
SAMPLE_RATIO = 0.1
SAMPLE_SEED = 42

# 结果只保存这一个CSV
SUMMARY_SAVE_PATH = "/root/autodl-tmp/japanese_entity_eval_10percent_summary.csv"

# 缓存不是结果文件，只是为了加速
ENTITY_CACHE_PATH = "/root/autodl-tmp/japanese_llm_entity_cache.json"

MAX_NEW_TOKENS = 512
MATCH_THRESHOLD = 0.55
STRICT_CATEGORY = True
STRICT_STATUS = True
EMPTY_BOTH_SCORE = 1.0
SAVE_CACHE_EVERY = 20

# =================================================


def normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text):
    text = normalize_text(text).lower()
    text = re.sub(r"[\s,，、。．.\(\)（）\[\]【】:：;；/／\-－−_]", "", text)
    return text


def text_hash(text):
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_item_uid(item, idx):
    return str(item.get("uid", idx))


def get_prediction_and_gt(item):
    if "predicted" in item and "ground truth" in item:
        return item["predicted"], item["ground truth"]

    if "best_impression" in item and "ground_truth" in item:
        return item["best_impression"], item["ground_truth"]

    if "final_pred" in item and "ground_truth" in item:
        return item["final_pred"], item["ground_truth"]

    raise KeyError("Cannot find prediction and ground-truth fields.")


def create_sampled_uids():
    with open(SAMPLE_SOURCE_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    all_uids = [
        get_item_uid(item, idx)
        for idx, item in enumerate(source_data)
    ]

    sample_size = max(1, int(round(len(all_uids) * SAMPLE_RATIO)))

    rng = random.Random(SAMPLE_SEED)
    sampled_uids = set(rng.sample(all_uids, sample_size))

    print(f"Sampled {len(sampled_uids)} / {len(all_uids)} cases.")
    return sampled_uids


def build_extraction_prompt(impression_text):
    impression_text = normalize_text(impression_text)

    return f"""
**Role**
You are an expert radiologist and a clinical information extraction system.

**Task**
Extract structured clinical entities from the following Japanese radiological Impression.

**Important**
- The input is Japanese.
- Extract information ONLY from the given Impression.
- Do NOT use the original Findings.
- Do NOT infer findings that are not explicitly stated.
- Output ONLY a valid JSON array. Do not include markdown, explanations, or extra text.

**Schema**
Each entity must be a JSON object with the following keys:
- category: one of Disease, Finding, Anatomy, Staging, Other.
- anatomy: anatomical location in Japanese if available, otherwise null.
- observation: clinical finding, diagnosis, abnormality, or staging item in Japanese.
- status: one of Present, Absent, Suspected.
- attributes: a list of Japanese modifiers, measurements, staging information, severity, or clinically relevant descriptors.

**Extraction Rules**
- Keep JSON keys in English.
- Keep clinical values in Japanese.
- Keep common medical abbreviations as written, such as MRI, DWI, CRM, EMVI, MRF, T1, T2, cT, cN, cM.
- If the expression contains 疑い, 疑われる, 可能性, 否定できない, set status to Suspected.
- If the expression contains 認めない, なし, 陰性, 否定的, or a minus sign such as EMVI -, set status to Absent.
- Otherwise, set status to Present.
- For cancer staging such as cT4N1, include it in attributes of the corresponding disease entity when possible.
- Do not create duplicate entities for the same clinical fact.

**Examples**

Input:
直腸癌（Rb、cT4N1）疑い

Output:
[
  {{
    "category": "Disease",
    "anatomy": "直腸Rb",
    "observation": "直腸癌",
    "status": "Suspected",
    "attributes": ["cT4N1"]
  }}
]

Input:
明らかな血管侵襲は認めません。少量腹水あり。

Output:
[
  {{
    "category": "Finding",
    "anatomy": null,
    "observation": "血管侵襲",
    "status": "Absent",
    "attributes": []
  }},
  {{
    "category": "Finding",
    "anatomy": "腹部",
    "observation": "腹水",
    "status": "Present",
    "attributes": ["少量"]
  }}
]

Input:
MRF陽性、EMVI陰性。CRM 0。

Output:
[
  {{
    "category": "Finding",
    "anatomy": null,
    "observation": "MRF",
    "status": "Present",
    "attributes": ["陽性"]
  }},
  {{
    "category": "Finding",
    "anatomy": null,
    "observation": "EMVI",
    "status": "Absent",
    "attributes": ["陰性"]
  }},
  {{
    "category": "Finding",
    "anatomy": null,
    "observation": "CRM",
    "status": "Absent",
    "attributes": ["0"]
  }}
]

**Current Impression**
{impression_text}

**JSON Output**
""".strip()


def load_model(model_path):
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def generate_text(tokenizer, model, prompt):
    chat_text = apply_chat_template(tokenizer, prompt)

    inputs = tokenizer(
        chat_text,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


def strip_thinking_text(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_json_array(raw_text):
    raw_text = strip_thinking_text(raw_text)
    raw_text = raw_text.strip()

    raw_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("entities"), list):
            return obj["entities"]
    except Exception:
        pass

    start = raw_text.find("[")
    end = raw_text.rfind("]")

    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw_text[start:end + 1])
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


def canonicalize_status(status):
    status = normalize_text(status).lower()

    if status in {"present", "positive", "あり", "有り", "陽性"}:
        return "Present"

    if status in {"absent", "negative", "なし", "無し", "陰性", "認めない"}:
        return "Absent"

    if status in {"suspected", "possible", "疑い", "疑われる", "可能性"}:
        return "Suspected"

    if "疑" in status or "可能" in status or "suspect" in status:
        return "Suspected"

    if "陰性" in status or "なし" in status or "absent" in status or "negative" in status:
        return "Absent"

    return "Present"


def canonicalize_category(category):
    category = normalize_text(category)
    allowed = {"Disease", "Finding", "Anatomy", "Staging", "Other"}

    if category in allowed:
        return category

    low = category.lower()

    if "disease" in low or "diagnosis" in low:
        return "Disease"

    if "finding" in low:
        return "Finding"

    if "anatomy" in low:
        return "Anatomy"

    if "stage" in low or "staging" in low:
        return "Staging"

    return "Other"


def clean_entity(entity):
    if not isinstance(entity, dict):
        return None

    category = canonicalize_category(entity.get("category", "Other"))

    anatomy = entity.get("anatomy", None)
    anatomy = normalize_text(anatomy) if anatomy not in [None, "null", "None", ""] else None

    observation = normalize_text(entity.get("observation", ""))
    if observation == "":
        return None

    status = canonicalize_status(entity.get("status", "Present"))

    attributes = entity.get("attributes", [])
    if attributes is None:
        attributes = []
    if isinstance(attributes, str):
        attributes = [attributes]
    if not isinstance(attributes, list):
        attributes = []

    clean_attrs = []
    seen = set()

    for attr in attributes:
        attr = normalize_text(attr)
        key = normalize_key(attr)
        if attr and key not in seen:
            seen.add(key)
            clean_attrs.append(attr)

    return {
        "category": category,
        "anatomy": anatomy,
        "observation": observation,
        "status": status,
        "attributes": clean_attrs
    }


def deduplicate_entities(entities):
    output = []
    seen = set()

    for entity in entities:
        if entity is None:
            continue

        key = (
            normalize_key(entity.get("category")),
            normalize_key(entity.get("anatomy")),
            normalize_key(entity.get("observation")),
            normalize_key(entity.get("status")),
            tuple(sorted(normalize_key(x) for x in entity.get("attributes", [])))
        )

        if key not in seen:
            seen.add(key)
            output.append(entity)

    return output


def extract_entities_with_llm(text, tokenizer, model, cache):
    text = normalize_text(text)
    key = text_hash(text)

    if key in cache:
        return cache[key]

    prompt = build_extraction_prompt(text)
    raw_output = generate_text(tokenizer, model, prompt)
    parsed = parse_json_array(raw_output)

    entities = []

    for entity in parsed:
        cleaned = clean_entity(entity)
        if cleaned is not None:
            entities.append(cleaned)

    entities = deduplicate_entities(entities)
    cache[key] = entities

    return entities


def char_ngrams(text, n=2):
    text = normalize_key(text)

    if not text:
        return set()

    if len(text) <= n:
        return {text}

    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_text(a, b):
    set_a = char_ngrams(a)
    set_b = char_ngrams(b)

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_list(a, b):
    a = a or []
    b = b or []

    set_a = {normalize_key(x) for x in a if normalize_key(x)}
    set_b = {normalize_key(x) for x in b if normalize_key(x)}

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def entity_similarity(pred_entity, gt_entity):
    if STRICT_CATEGORY:
        if normalize_key(pred_entity.get("category")) != normalize_key(gt_entity.get("category")):
            return 0.0

    if STRICT_STATUS:
        if normalize_key(pred_entity.get("status")) != normalize_key(gt_entity.get("status")):
            return 0.0

    obs_score = jaccard_text(
        pred_entity.get("observation"),
        gt_entity.get("observation")
    )

    anatomy_score = jaccard_text(
        pred_entity.get("anatomy"),
        gt_entity.get("anatomy")
    )

    attr_score = jaccard_list(
        pred_entity.get("attributes"),
        gt_entity.get("attributes")
    )

    return 0.60 * obs_score + 0.25 * anatomy_score + 0.15 * attr_score


def match_entities(pred_entities, gt_entities):
    if not pred_entities and not gt_entities:
        return EMPTY_BOTH_SCORE, EMPTY_BOTH_SCORE, EMPTY_BOTH_SCORE

    if not pred_entities:
        return 0.0, 0.0, 0.0

    if not gt_entities:
        return 0.0, 0.0, 0.0

    score_matrix = []

    for pred_entity in pred_entities:
        row = []
        for gt_entity in gt_entities:
            row.append(entity_similarity(pred_entity, gt_entity))
        score_matrix.append(row)

    match_count = 0

    if linear_sum_assignment is not None and np is not None:
        matrix = np.array(score_matrix, dtype=float)
        row_indices, col_indices = linear_sum_assignment(-matrix)

        for row, col in zip(row_indices, col_indices):
            if float(matrix[row, col]) >= MATCH_THRESHOLD:
                match_count += 1
    else:
        pairs = []

        for i, row in enumerate(score_matrix):
            for j, score in enumerate(row):
                if score >= MATCH_THRESHOLD:
                    pairs.append((score, i, j))

        pairs.sort(reverse=True)

        used_pred = set()
        used_gt = set()

        for score, i, j in pairs:
            if i in used_pred or j in used_gt:
                continue

            used_pred.add(i)
            used_gt.add(j)
            match_count += 1

    precision = match_count / len(pred_entities)
    recall = match_count / len(gt_entities)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def evaluate_one_file(data_path, tokenizer, model, cache, sampled_uids):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indexed_data = [
        (idx, item)
        for idx, item in enumerate(data)
        if get_item_uid(item, idx) in sampled_uids
    ]

    precision_list = []
    recall_list = []
    f1_list = []

    valid_count = 0

    for idx, item in enumerate(tqdm(indexed_data, desc=os.path.basename(data_path))):
        original_idx, each_item = item

        try:
            pred_text, gt_text = get_prediction_and_gt(each_item)
        except Exception as e:
            print(f"Skip item {original_idx} in {data_path} because of missing fields: {e}")
            continue

        pred_entities = extract_entities_with_llm(
            pred_text,
            tokenizer,
            model,
            cache
        )

        gt_entities = extract_entities_with_llm(
            gt_text,
            tokenizer,
            model,
            cache
        )

        precision, recall, f1 = match_entities(
            pred_entities,
            gt_entities
        )

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        valid_count += 1

        if valid_count % SAVE_CACHE_EVERY == 0:
            save_json(cache, ENTITY_CACHE_PATH)

    return {
        "filename": os.path.basename(data_path),
        "file": data_path,
        "num_total": len(data),
        "num_sampled": len(indexed_data),
        "num_valid": valid_count,
        "precision": mean(precision_list) if precision_list else 0.0,
        "recall": mean(recall_list) if recall_list else 0.0,
        "f1": mean(f1_list) if f1_list else 0.0
    }


def save_summary_csv(summary, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "filename",
        "file",
        "num_total",
        "num_sampled",
        "num_valid",
        "precision",
        "recall",
        "f1"
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary:
            writer.writerow({
                "filename": row["filename"],
                "file": row["file"],
                "num_total": row["num_total"],
                "num_sampled": row["num_sampled"],
                "num_valid": row["num_valid"],
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}",
                "f1": f"{row['f1']:.6f}"
            })


def main():
    tokenizer, model = load_model(MODEL_PATH)
    cache = load_json(ENTITY_CACHE_PATH, {})

    sampled_uids = create_sampled_uids()

    summary = []

    for data_path in DATA_PATHS:
        if not os.path.exists(data_path):
            print(f"File not found, skip: {data_path}")
            continue

        print(f"\nEvaluating: {data_path}")

        result = evaluate_one_file(
            data_path=data_path,
            tokenizer=tokenizer,
            model=model,
            cache=cache,
            sampled_uids=sampled_uids
        )

        summary.append(result)
        save_json(cache, ENTITY_CACHE_PATH)

        print(
            f"{result['filename']} | "
            f"Sampled={result['num_sampled']} | "
            f"P={result['precision']:.4f}, "
            f"R={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}"
        )

    save_summary_csv(summary, SUMMARY_SAVE_PATH)

    print("\n--- Final Summary ---")
    for row in summary:
        print(
            f"{row['filename']} | "
            f"Sampled={row['num_sampled']} | "
            f"P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}"
        )

    print(f"\nSummary saved to: {SUMMARY_SAVE_PATH}")
    print(f"Entity cache saved to: {ENTITY_CACHE_PATH}")


if __name__ == "__main__":
    main()