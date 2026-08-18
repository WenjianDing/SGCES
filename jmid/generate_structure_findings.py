# coding=utf-8
import json
import os
import textwrap
import time

from openai import OpenAI
from tqdm import tqdm


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = "qwen3.6-27b"

TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH",
    "/root/autodl-tmp/jmid_test_data.json",
)
SAVE_DATA_PATH = os.getenv(
    "SAVE_DATA_PATH",
    "/root/autodl-tmp/jmid_gt_findings_structure_qwen.json",
)

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.8"))
TOP_K = int(os.getenv("TOP_K", "20"))
MIN_P = float(os.getenv("MIN_P", "0.0"))
REQUEST_RETRY = int(os.getenv("REQUEST_RETRY", "3"))


FINDINGS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "anatomy": {"type": ["string", "null"]},
                    "observation": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Present", "Absent", "Suspected"],
                    },
                    "attributes": {"type": ["string", "null"]},
                },
                "required": [
                    "category",
                    "anatomy",
                    "observation",
                    "status",
                    "attributes",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)


def build_prompt(findings):
    prompt_raw = f"""
        #Role#
        You are a senior radiologist and medical data structuring expert.
        Your task is to transform Japanese MRI radiology "Findings" text into a high-precision structured JSON format for downstream retrieval and reasoning.

        #Language Requirement#
        The JSON keys must remain in English exactly as specified: `category`, `anatomy`, `observation`, `status`, and `attributes`.
        The values for `category`, `anatomy`, `observation`, and `attributes` must be written in Japanese.
        Do not translate Japanese medical findings into English.
        For code compatibility, the value of `status` must be one of the English labels: `Present`, `Absent`, or `Suspected`.

        #Task#
        Carefully read the input Japanese MRI "Findings" text.
        Extract all clinically meaningful observations, including positive findings, explicitly stated negative findings, suspected findings, measurements, anatomical locations, contrast enhancement, DWI signal, staging-related descriptions, lymph nodes, ascites, vascular invasion, technical/protocol information, and comparison or addendum information when clinically relevant.
        Do not omit clinically valuable details such as size, extent, invasion, laterality, severity, morphology, or uncertainty.

        #Schema Definition#
        Strictly output a JSON object containing a `findings` list.
        Each item in the list must adhere to the following dictionary structure:
        {{
          "category": "Japanese category label, e.g., 腫瘍, リンパ節, 血管, 腹水, 撮像条件, 治療後変化, その他",
          "anatomy": "Specific anatomical location in Japanese, e.g., 直腸Rb, 右側壁, 肛門挙筋, 坐骨直腸窩, 上直腸動脈周囲. If not specified, set to null.",
          "observation": "Core clinical observation in Japanese, e.g., 腫瘤, 壁欠損, 筋層外進展, リンパ節腫大, 腹水, 血管侵襲なし.",
          "status": "Enum: [Present, Absent, Suspected]",
          "attributes": "Detailed modifiers in Japanese, such as size, signal, enhancement, border, morphology, stage, uncertainty, or protocol details. If none, set to null."
        }}

        #Extraction Rules#
        1. Extract positive findings and mark their `status` as `Present`.
        2. Extract explicit negative findings and mark their `status` as `Absent`.
           Examples: 「認めません」, 「なし」, 「-」, 「陰性」.
        3. Extract uncertain or suspicious findings and mark their `status` as `Suspected`.
           Examples: 「疑い」, 「思われます」, 「可能性」, 「否定できません」.
        4. Keep all clinically important numeric measurements and staging descriptors in Japanese text, including units such as cm, mm, or Japanese unit symbols.
        5. Keep Japanese anatomical expressions as Japanese values. Do not romanize them unless the source text itself uses Roman characters.
        6. If the source text contains English abbreviations such as DWI, CRM, EMVI, MRF, T1, T2, or FIESTA, keep them as written inside the Japanese value.
        7. You must return RAW JSON only. Do NOT wrap the output in Markdown code blocks. Do not include any introductory, explanatory, or conversational text.

        #Input Findings#
        {findings}
        """
    return textwrap.dedent(prompt_raw).strip()


def get_result(full_prompt):
    messages = [{"role": "user", "content": full_prompt}]
    last_error = None

    for attempt in range(1, REQUEST_RETRY + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                extra_body={
                    "top_k": TOP_K,
                    "min_p": MIN_P,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "structured_outputs": {"json": FINDINGS_JSON_SCHEMA},
                },
            )
            prediction = completion.choices[0].message.content or ""
            # print("prediction")
            # print(prediction)
            return prediction
        except Exception as exc:
            last_error = exc
            print(f"[Warning] Request failed, attempt {attempt}/{REQUEST_RETRY}: {exc}")
            time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"vLLM request failed after {REQUEST_RETRY} attempts") from last_error


def parse_and_align_json(raw_text):
    required_keys = ["category", "anatomy", "observation", "status", "attributes"]
    fallback_result = {"findings": []}

    clean_text = raw_text.strip()
    if "```json" in clean_text:
        clean_text = clean_text.split("```json", 1)[1]
    if "```" in clean_text:
        clean_text = clean_text.split("```", 1)[0]
    clean_text = clean_text.strip()

    try:
        parsed_data = json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"\n[Warning] JSON Decode Error. Raw text:\n{raw_text}\n")
        return fallback_result

    if isinstance(parsed_data, dict):
        findings_list = parsed_data.get("findings", [])
    elif isinstance(parsed_data, list):
        findings_list = parsed_data
    else:
        findings_list = []

    aligned_findings = []
    for item in findings_list:
        if isinstance(item, dict):
            aligned_item = {key: item.get(key, None) for key in required_keys}
            if aligned_item["status"] not in {"Present", "Absent", "Suspected"}:
                aligned_item["status"] = "Suspected"
            aligned_findings.append(aligned_item)

    return {"findings": aligned_findings}


def load_progress(save_data_path):
    if not os.path.exists(save_data_path):
        print("[*] No existing progress file found. Starting from scratch...")
        return {}

    try:
        with open(save_data_path, "r", encoding="utf-8") as f_save:
            id2findings = json.load(f_save)
        print(f"[*] Found existing progress with {len(id2findings)} records. Resuming...")
        return id2findings
    except json.JSONDecodeError:
        print("[!] Existing progress file is broken or empty. Restarting from scratch.")
        return {}


def main():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f1:
        test_data = json.load(f1)

    id2findings = load_progress(SAVE_DATA_PATH)

    for each_ct_data in tqdm(test_data):
        uid = str(each_ct_data["uid"])
        if uid in id2findings:
            continue

        prompt = build_prompt(each_ct_data["findings"])
        baseline_result = get_result(prompt)
        structured_output = parse_and_align_json(baseline_result)
        id2findings[uid] = structured_output

        with open(SAVE_DATA_PATH, "w", encoding="utf-8") as outfile:
            json.dump(id2findings, outfile, ensure_ascii=False, indent=4)

    print("\n[*] All data processing completed.")


if __name__ == "__main__":
    main()
