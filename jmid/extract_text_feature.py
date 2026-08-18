# coding=utf-8
import json
import os
import pickle

import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


MODEL_NAME = r"...\bge-m3"
JSON_FILE_PATH = r"C:\Work\pricai\jmid\jmid_test\jmid_test_data.json"
SAVE_PATH = "jmid_uid_to_text_features_bgem3.pkl"

BATCH_SIZE = 12
MAX_LENGTH = 8192
USE_FP16 = True


def load_bgem3_model():
    """
    BGE-M3 official dense embedding usage:
      model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
      model.encode(texts, batch_size=..., max_length=...)["dense_vecs"]
    """
    return BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def extract_findings_embeddings(json_path, batch_size=BATCH_SIZE):
    """
    Read {uid, findings, impression} JSON records and extract Japanese findings
    embeddings with BGE-M3. Returns {uid: embedding}.
    """
    print(f"Loading BGE-M3 model: {MODEL_NAME}")
    model = load_bgem3_model()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded JSON file with {len(data)} records.")

    uid_to_embedding = {}

    for i in tqdm(range(0, len(data), batch_size), desc="Encoding raw Findings with BGE-M3"):
        batch_data = data[i:i + batch_size]

        valid_texts = []
        valid_uids = []

        for item in batch_data:
            uid = str(item.get("uid", "unknown"))
            findings_text = item.get("findings")

            if findings_text is None or str(findings_text).strip() == "":
                continue

            # BGE-M3 can handle long multilingual reports. Keep the original
            # Japanese text and let max_length control truncation if needed.
            valid_texts.append(str(findings_text).strip())
            valid_uids.append(uid)

        if not valid_texts:
            continue

        output = model.encode(
            valid_texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
        )
        features_np = np.asarray(output["dense_vecs"], dtype=np.float32)

        # Existing retrieval code uses np.dot as cosine similarity, so store
        # L2-normalized vectors.
        features_np = l2_normalize(features_np).astype(np.float32)

        for idx, uid in enumerate(valid_uids):
            uid_to_embedding[uid] = features_np[idx]

    return uid_to_embedding


if __name__ == "__main__":
    text_result_dict = extract_findings_embeddings(JSON_FILE_PATH, BATCH_SIZE)

    print(f"\nDone. Successfully processed {len(text_result_dict)} Findings records.")

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(text_result_dict, f)

    print(f"Saved text feature dictionary to: {SAVE_PATH}")

    if "1" in text_result_dict:
        print("Example UID '1' vector shape:", text_result_dict["1"].shape)
        print("Example UID '1' vector:", text_result_dict["1"])
