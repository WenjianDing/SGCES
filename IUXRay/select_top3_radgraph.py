# coding=utf-8
"""
radgraph_inspired_retrieval.py
==============================
基于 RadGraph 思想的实体集合检索，使用 GPT-4 抽取的结构化 findings。

核心思路：
  - 将每个 finding 转为 (Anatomy, Observation, Status) 三元组（对应 RadGraph 实体）
  - 用实体集合 F1（Precision/Recall/F1）作为相似度分数（RadGraph 官方评估方式）
  - 用 F1 分数检索 Top-K 最相似样本

输入：gt_findings_structure_gpt54_dict.json
输出：radgraph_sim_mapping_top3.json

依赖安装：无额外依赖
"""

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ══════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════
STRUCT_PATH  = "C:/Work/pricai/gt_findings_structure_gpt54_dict.json"
OUTPUT_PATH  = "C:/Work/pricai/radgraph_top3.json"
TOP_K        = 3

# Status 权重：阳性 finding 比阴性更有区分价值（参考 RadGraph-F1 设计）
STATUS_WEIGHT = {"Present": 1.0, "Absent": 1.0}
# ══════════════════════════════════════════════════════════


def to_entity_set(obs_list: list) -> dict:
    """
    将结构化 findings 列表转为 RadGraph 风格的加权实体集合。

    对应关系：
      RadGraph Anatomy          → Anatomy 字段
      RadGraph Obs:Present      → Status=Present 的 Observation
      RadGraph Obs:Absent       → Status=Absent  的 Observation
      RadGraph located_at 关系  → (Anatomy, Observation) 绑定

    返回：
      {
        entity_key(str): weight(float)
      }
      entity_key = "Anatomy||Observation||Status"
    """
    entities = {}
    for obs in obs_list:
        anatomy    = (obs.get("Anatomy") or "").strip().lower()
        observation = (obs.get("Observation") or "").strip().lower()
        status     = obs.get("Status", "Present")
        category   = (obs.get("Category") or "").strip()

        # 跳过空条目
        if not observation:
            continue

        # 构造实体 key（类似 RadGraph 的实体+关系）
        entity_key = f"{category}||{anatomy}||{observation}||{status}"
        weight = STATUS_WEIGHT.get(status, 1.0)
        entities[entity_key] = weight

    return entities


def radgraph_f1(q_entities: dict, c_entities: dict) -> float:
    """
    计算两个实体集合之间的加权 F1 分数（RadGraph-F1 核心逻辑）。

    Precision = 匹配权重之和 / candidate 总权重
    Recall    = 匹配权重之和 / query 总权重
    F1        = 2 * P * R / (P + R)

    完全匹配：entity_key 完全相同
    部分匹配：anatomy 和 status 相同但 observation 词汇有重叠（软匹配）
    """
    if not q_entities or not c_entities:
        return 0.0

    def token_sim(a: str, b: str) -> float:
        """词级别 Jaccard"""
        a_t = set(a.lower().split())
        b_t = set(b.lower().split())
        if not a_t or not b_t:
            return 0.0
        return len(a_t & b_t) / len(a_t | b_t)

    # 计算匹配权重（每个 query 实体找最佳匹配的 candidate 实体）
    matched_weight = 0.0
    for q_key, q_w in q_entities.items():
        q_parts = q_key.split("||")  # [category, anatomy, observation, status]
        if len(q_parts) != 4:
            continue
        q_cat, q_anat, q_obs, q_stat = q_parts

        best_match = 0.0
        for c_key, c_w in c_entities.items():
            c_parts = c_key.split("||")
            if len(c_parts) != 4:
                continue
            c_cat, c_anat, c_obs, c_stat = c_parts

            # 硬约束：Category 和 Status 必须一致
            if q_cat != c_cat or q_stat != c_stat:
                continue

            # 软匹配：observation 词汇重叠
            obs_sim  = token_sim(q_obs, c_obs)
            anat_sim = token_sim(q_anat, c_anat)
            match_score = 0.6 * obs_sim + 0.4 * anat_sim

            if match_score > best_match:
                best_match = match_score

        matched_weight += q_w * best_match

    # Precision：candidate 覆盖了多少 query 实体
    q_total = sum(q_entities.values())
    c_total = sum(c_entities.values())

    if q_total == 0 or c_total == 0:
        return 0.0

    # 双向匹配：同时计算 query→candidate 和 candidate→query
    # （对应 RadGraph-F1 的 Precision 和 Recall）
    recall    = matched_weight / q_total

    # candidate→query 方向
    matched_weight_rev = 0.0
    for c_key, c_w in c_entities.items():
        c_parts = c_key.split("||")
        if len(c_parts) != 4:
            continue
        c_cat, c_anat, c_obs, c_stat = c_parts

        best_match = 0.0
        for q_key, q_w in q_entities.items():
            q_parts = q_key.split("||")
            if len(q_parts) != 4:
                continue
            q_cat, q_anat, q_obs, q_stat = q_parts

            if c_cat != q_cat or c_stat != q_stat:
                continue

            obs_sim  = token_sim(c_obs, q_obs)
            anat_sim = token_sim(c_anat, q_anat)
            match_score = 0.6 * obs_sim + 0.4 * anat_sim

            if match_score > best_match:
                best_match = match_score

        matched_weight_rev += c_w * best_match

    precision = matched_weight_rev / c_total

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def main():
    # 1. 加载结构化数据
    print(f"Loading structured data from {STRUCT_PATH} ...")
    with open(STRUCT_PATH, "r", encoding="utf-8") as f:
        struct_data = json.load(f)

    uids = list(struct_data.keys())
    print(f"  {len(uids)} samples loaded.")

    # 2. 预计算所有样本的实体集合
    print("Converting to entity sets ...")
    entity_sets = {}
    for uid in tqdm(uids, desc="Building entity sets"):
        entity_sets[uid] = to_entity_set(struct_data[uid])

    # 3. 计算所有样本对之间的 RadGraph-F1 相似度并检索 Top-K
    print(f"Computing RadGraph-F1 similarity and retrieving Top-{TOP_K} ...")
    results = {}

    for i, q_uid in enumerate(tqdm(uids, desc="Retrieving")):
        q_entities = entity_sets[q_uid]
        scores = []

        for j, c_uid in enumerate(uids):
            if c_uid == q_uid:
                scores.append(-1.0)   # 排除自身
                continue
            f1 = radgraph_f1(q_entities, entity_sets[c_uid])
            scores.append(f1)

        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        results[q_uid] = [uids[idx] for idx in top_indices]

    # 4. 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完成！结果保存至: {OUTPUT_PATH}")
    print(f"   共 {len(results)} 条记录，每条 Top-{TOP_K}。")

    # 5. 打印示例
    print("\n示例输出（前3条）：")
    for uid, top_uids in list(results.items())[:3]:
        print(f"  uid={uid} → {top_uids}")

    # 6. 打印一个样本的实体集合示例
    sample_uid = uids[0]
    print(f"\n样本 uid={sample_uid} 的实体集合：")
    for k, w in list(entity_sets[sample_uid].items())[:5]:
        print(f"  [{w:.1f}] {k}")


if __name__ == "__main__":
    main()