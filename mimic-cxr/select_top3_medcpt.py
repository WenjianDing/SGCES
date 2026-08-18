# coding=utf-8
"""
medcpt_retrieval.py
===================
使用 MedCPT 对 findings 文本进行检索，为每个样本找出最相似的 Top-3 样本。

MedCPT 使用双编码器架构：
  - Query Encoder:   ncats/MedCPT-Query-Encoder   （编码查询文本）
  - Article Encoder: ncats/MedCPT-Article-Encoder  （编码候选文本）

依赖安装：
  pip install torch transformers tqdm

输入：ct_data.json  （list，每个元素含 uid / findings 字段）
输出：medcpt_sim_mapping_top3.json  （{uid: [uid1, uid2, uid3]}）
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ══════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════
DATA_PATH   = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"
OUTPUT_PATH      = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_medcpt_top3.json"
TOP_K            = 3
BATCH_SIZE       = 128
MAX_LENGTH       = 512
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_MODEL_NAME = "C:/Users/81163/Downloads/MedCPT-Query-Encoder"
DOC_MODEL_NAME   = "C:/Users/81163/Downloads/MedCPT-Article-Encoder"
# ══════════════════════════════════════════════════════════


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list, desc: str) -> np.ndarray:
    """
    批量编码文本列表，返回 L2 归一化后的 (N, d) float32 numpy 数组。
    使用 [CLS] token 的输出作为句子表示（MedCPT 官方做法）。
    """
    all_vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        output = model(**encoded)
        # MedCPT 使用 [CLS] token（last_hidden_state 的第0位）
        vecs = output.last_hidden_state[:, 0, :]   # (B, d)
        vecs = F.normalize(vecs, dim=-1)
        all_vecs.append(vecs.cpu().float().numpy())

    return np.vstack(all_vecs).astype("float32")


def main():
    # 1. 加载数据
    print(f"Loading data from {DATA_PATH} ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    uids     = [str(item["uid"]) for item in data]
    findings = [item.get("findings", "") for item in data]
    print(f"  {len(uids)} samples loaded.")

    # 2. 加载 Query Encoder（用于编码每个查询样本的 findings）
    print(f"\nLoading Query Encoder: {QUERY_MODEL_NAME} ...")
    q_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
    q_model     = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(DEVICE).eval()

    # 3. 加载 Article Encoder（用于编码候选库的 findings）
    print(f"Loading Article Encoder: {DOC_MODEL_NAME} ...")
    d_tokenizer = AutoTokenizer.from_pretrained(DOC_MODEL_NAME)
    d_model     = AutoModel.from_pretrained(DOC_MODEL_NAME).to(DEVICE).eval()

    # 4. 编码所有 findings 文本
    #    query_vecs:  每个样本作为 query 时的向量  (N, d)
    #    doc_vecs:    每个样本作为候选时的向量     (N, d)
    print("\nEncoding findings as queries ...")
    query_vecs = encode_texts(q_model, q_tokenizer, findings,
                              desc="Query encoding")

    print("Encoding findings as documents ...")
    doc_vecs = encode_texts(d_model, d_tokenizer, findings,
                            desc="Doc encoding")

    # 5. 批量计算相似度并检索 Top-K
    #    sim_matrix[i, j] = cosine_sim(query_vecs[i], doc_vecs[j])
    #    向量已 L2 归一化，所以内积 = cosine similarity
    print(f"\nRetrieving Top-{TOP_K} for each sample ...")
    results = {}

    # 分批计算避免 OOM（整个矩阵 N×N 如果 N 很大会爆显存）
    CHUNK = 256
    sim_matrix = np.zeros((len(uids), len(uids)), dtype="float32")

    for i in tqdm(range(0, len(uids), CHUNK), desc="Computing similarity"):
        q_chunk = torch.tensor(query_vecs[i:i+CHUNK]).to(DEVICE)   # (chunk, d)
        d_all   = torch.tensor(doc_vecs).to(DEVICE)                 # (N, d)
        sims    = torch.mm(q_chunk, d_all.T).cpu().numpy()          # (chunk, N)
        sim_matrix[i:i+CHUNK] = sims

    # 6. 对每个样本取 Top-K，排除自身
    for i, uid in enumerate(uids):
        scores = sim_matrix[i].copy()
        scores[i] = -np.inf                              # 排除自身
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        results[uid] = [uids[idx] for idx in top_indices]

    # 7. 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完成！结果保存至: {OUTPUT_PATH}")
    print(f"   共 {len(results)} 条记录，每条 Top-{TOP_K} 个相似样本。")

    # 8. 打印示例
    print("\n示例输出（前3条）：")
    for uid, top_uids in list(results.items())[:3]:
        print(f"  uid={uid} → {top_uids}")


if __name__ == "__main__":
    main()