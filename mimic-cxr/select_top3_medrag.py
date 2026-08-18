# coding=utf-8
"""
medrag_retrieval_all_in_one.py
==============================
一步完成 MedRAG 检索（Xiong et al., ACL 2024）：
  Step 1: BM25   检索 Top-N
  Step 2: MedCPT 检索 Top-N
  Step 3: RRF 融合 → 最终 Top-3

依赖安装：
  pip install rank_bm25 torch transformers tqdm

输入：ct_data.json
输出：medrag_sim_mapping_top3.json
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

# ══════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════
DATA_PATH   = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"
OUTPUT_PATH      = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_medrag_top3.json"

TOP_K            = 3      # 最终返回的样本数
RRF_N            = 50     # BM25 和 MedCPT 各自先取 Top-N 再融合
RRF_K            = 60     # RRF 标准超参数

BATCH_SIZE       = 128
MAX_LENGTH       = 512
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_MODEL_NAME = "C:/Users/81163/Downloads/MedCPT-Query-Encoder"
DOC_MODEL_NAME   = "C:/Users/81163/Downloads/MedCPT-Article-Encoder"
# ══════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────
#  Step 1: BM25 检索
# ──────────────────────────────────────────────────────────
def bm25_retrieve(uids, findings, top_n):
    """返回 {uid: [uid1, uid2, ...]} Top-N 排序列表"""
    print("Building BM25 index ...")
    tokenized = [f.lower().split() for f in findings]
    bm25 = BM25Okapi(tokenized)

    results = {}
    for i, uid in enumerate(tqdm(uids, desc="BM25 retrieving")):
        scores = bm25.get_scores(tokenized[i])
        scores[i] = -np.inf                          # 排除自身
        top_indices = np.argsort(scores)[::-1][:top_n]
        results[uid] = [uids[j] for j in top_indices]

    return results


# ──────────────────────────────────────────────────────────
#  Step 2: MedCPT 检索
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def encode_texts(model, tokenizer, texts, desc):
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
        vecs = model(**encoded).last_hidden_state[:, 0, :]  # [CLS]
        vecs = F.normalize(vecs, dim=-1)
        all_vecs.append(vecs.cpu().float().numpy())
    return np.vstack(all_vecs).astype("float32")


def medcpt_retrieve(uids, findings, top_n):
    """返回 {uid: [uid1, uid2, ...]} Top-N 排序列表"""
    print(f"\nLoading MedCPT encoders on {DEVICE} ...")
    q_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
    q_model     = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(DEVICE).eval()
    d_tokenizer = AutoTokenizer.from_pretrained(DOC_MODEL_NAME)
    d_model     = AutoModel.from_pretrained(DOC_MODEL_NAME).to(DEVICE).eval()

    query_vecs = encode_texts(q_model, q_tokenizer, findings, "MedCPT query encoding")
    doc_vecs   = encode_texts(d_model, d_tokenizer, findings, "MedCPT doc encoding")

    # 分块计算相似度矩阵，避免 OOM
    print("Computing MedCPT similarity matrix ...")
    CHUNK = 256
    results = {}

    for i in tqdm(range(0, len(uids), CHUNK), desc="MedCPT retrieving"):
        q_chunk = torch.tensor(query_vecs[i:i+CHUNK]).to(DEVICE)
        d_all   = torch.tensor(doc_vecs).to(DEVICE)
        sims    = torch.mm(q_chunk, d_all.T).cpu().numpy()   # (chunk, N)

        for local_j, global_i in enumerate(range(i, min(i+CHUNK, len(uids)))):
            scores = sims[local_j].copy()
            scores[global_i] = -np.inf                       # 排除自身
            top_indices = np.argsort(scores)[::-1][:top_n]
            results[uids[global_i]] = [uids[k] for k in top_indices]

    return results


# ──────────────────────────────────────────────────────────
#  Step 3: RRF 融合
# ──────────────────────────────────────────────────────────
def rrf_fusion(ranked_lists, k=60):
    """多个排序列表 → RRF 融合后的排序列表"""
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, uid in enumerate(ranked_list, start=1):
            scores[uid] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda u: scores[u], reverse=True)


# ──────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────
def main():
    # 1. 加载数据
    print(f"Loading data from {DATA_PATH} ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    uids     = [str(item["uid"]) for item in data]
    findings = [item.get("findings", "") for item in data]
    print(f"  {len(uids)} samples loaded.\n")

    # 2. BM25 检索
    bm25_results = bm25_retrieve(uids, findings, top_n=RRF_N)

    # 3. MedCPT 检索
    medcpt_results = medcpt_retrieve(uids, findings, top_n=RRF_N)

    # 4. RRF 融合
    print("\nApplying RRF fusion ...")
    final_results = {}
    for uid in tqdm(uids, desc="RRF"):
        bm25_list   = bm25_results.get(uid, [])
        medcpt_list = medcpt_results.get(uid, [])

        fused = rrf_fusion([bm25_list, medcpt_list], k=RRF_K)
        fused = [u for u in fused if u != uid]      # 排除自身（保险）
        final_results[uid] = fused[:TOP_K]

    # 5. 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完成！结果保存至: {OUTPUT_PATH}")
    print(f"   共 {len(final_results)} 条，每条 Top-{TOP_K}。")

    print("\n示例输出（前3条）：")
    for uid, top_uids in list(final_results.items())[:3]:
        print(f"  uid={uid} → {top_uids}")


if __name__ == "__main__":
    main()