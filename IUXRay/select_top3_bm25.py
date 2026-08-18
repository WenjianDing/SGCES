# coding=utf-8
"""
bm25_retrieval.py
=================
使用 BM25 对 findings 文本进行检索，为每个样本找出最相似的 Top-3 样本。

依赖安装：
  pip install rank_bm25

输入：ct_data.json  （list，每个元素含 uid / findings 字段）
输出：bm25_sim_mapping_top3.json  （{uid: [uid1, uid2, uid3]}）
"""

import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# ══════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════
DATA_PATH   = r"C:\Work\pricai\ct_data.json"
OUTPUT_PATH = r"C:\Work\pricai\bm25_top3.json"
TOP_K       = 3
# ══════════════════════════════════════════════════════════


def simple_tokenize(text: str) -> list:
    """
    简单小写+分词，去掉空token。
    如需更好效果可换成 nltk.word_tokenize。
    """
    if not text:
        return ["<empty>"]
    return [w for w in text.lower().split() if w]


def main():
    # 1. 加载数据
    print(f"Loading data from {DATA_PATH} ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    uids     = [str(item["uid"]) for item in data]
    findings = [item.get("findings", "") for item in data]
    print(f"  {len(uids)} samples loaded.")

    # 2. 构建 BM25 索引
    print("Building BM25 index ...")
    tokenized_corpus = [simple_tokenize(f) for f in findings]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. 逐样本检索
    print(f"Retrieving Top-{TOP_K} for each sample ...")
    results = {}

    for i, (uid, finding) in enumerate(tqdm(zip(uids, findings), total=len(uids))):
        query_tokens = simple_tokenize(finding)
        scores = bm25.get_scores(query_tokens)   # shape: (N,)

        # 排除自身
        scores[i] = -np.inf

        top_indices = np.argsort(scores)[::-1][:TOP_K]
        results[uid] = [uids[idx] for idx in top_indices]

    # 4. 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完成！结果保存至: {OUTPUT_PATH}")
    print(f"   共 {len(results)} 条记录，每条 Top-{TOP_K} 个相似样本。")

    # 5. 打印几条示例
    print("\n示例输出（前3条）：")
    for uid, top_uids in list(results.items())[:3]:
        print(f"  uid={uid} → {top_uids}")


if __name__ == "__main__":
    main()