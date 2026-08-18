import numpy as np
import pickle
import json
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def calculate_cosine_similarity(vec_q, matrix_db):
    """计算余弦相似度 (假设已 L2 归一化)"""
    return np.dot(matrix_db, vec_q)


def token_overlap(a, b):
    """词级别 Jaccard 相似度，处理 null/None 字符串"""
    # 处理空值
    a_empty = (not a or a.strip().lower() in ("none", "null", ""))
    b_empty = (not b or b.strip().lower() in ("none", "null", ""))
    if a_empty and b_empty:
        return 1.0
    if a_empty or b_empty:
        return 0.5
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def safe_token_overlap(str1, str2):
    """安全处理 None 值的 token overlap"""
    # 双方都没提及这个属性，说明在这个维度上达到了"默契的完美一致"
    if not str1 and not str2:
        return 1.0
    # 一方有一方没有，完全不匹配
    if bool(str1) != bool(str2):
        return 0.0
    # 这里调用你原来的 token_overlap 逻辑，或者简单的集合交并比(Jaccard)
    return token_overlap(str1, str2)


def match_single_obs(q_obs, c_obs):
    """
    单个 observation 之间的层次化匹配分数，返回 0~1。

    层次：
      1. Category 硬匹配 → 不同则直接返回 0
      2. Status   硬匹配 → 不同则直接返回 0
      3. Observation 软匹配（词重叠），权重 0.5
      4. Anatomy     软匹配（词重叠），权重 0.3
      5. Attributes  软匹配（词重叠），权重 0.2
    """
    # 硬匹配：Category
    if q_obs.get("Category") != c_obs.get("Category"):
        return 0.0

    # 硬匹配：Status（Present vs Absent 是本质差异）
    if q_obs.get("Status") != c_obs.get("Status"):
        return 0.0

    # 软匹配：Observation（最重要的语义字段）
    obs_score  = token_overlap(q_obs.get("Observation"), c_obs.get("Observation"))

    # 软匹配：Anatomy
    anat_score = token_overlap(q_obs.get("Anatomy"), c_obs.get("Anatomy"))

    # 软匹配：Attributes
    attr_score = token_overlap(q_obs.get("Attributes"), c_obs.get("Attributes"))

    return 0.5 * obs_score + 0.3 * anat_score + 0.2 * attr_score


def symbolic_struct_sim(q_obs_list, c_obs_list):
    """
    用 Hungarian 算法对 query 的 n 个 obs 和 candidate 的 m 个 obs
    做最优匹配，返回 0~1 的相似度分数。

    归一化方式：matched 总分 / n（query obs 数量）
    未匹配的 query obs 计 0 分，体现"漏报"的惩罚。
    """
    n = len(q_obs_list)
    m = len(c_obs_list)

    if n == 0 or m == 0:
        return 0.0

    # 构造 n×m 得分矩阵
    score_matrix = np.zeros((n, m))
    for i, q_obs in enumerate(q_obs_list):
        for j, c_obs in enumerate(c_obs_list):
            score_matrix[i, j] = match_single_obs(q_obs, c_obs)

    # Hungarian 最优匹配（最大化总分）
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    matched_scores = score_matrix[row_ind, col_ind]

    # 除以 query obs 数量归一化
    return float(matched_scores.sum() / n)




def hierarchical_multimodal_retrieval(img_features, txt_features, struct_data,
                                      alpha=0.2, beta=0.2, gamma=0.6,
                                      top_n_coarse=50, top_k_fine=3):
    """
    层次化三模态检索（符号化结构匹配版）

    alpha: 图像权重
    beta:  文本权重
    gamma: 结构化符号匹配权重
    约束:  alpha + beta + gamma = 1
    """
    uids = list(img_features.keys())
    uid_array = np.array(uids)

    img_matrix = np.array([img_features[uid] for uid in uids])
    txt_matrix = np.array([txt_features[uid] for uid in uids])

    final_sim_mapping = {}

    for q_uid in tqdm(uids, desc="三模态符号化检索中"):
        q_img    = img_features[q_uid]
        q_txt    = txt_features[q_uid]
        q_struct = struct_data.get(str(q_uid), [])

        # ── 步骤 1：粗筛（图像 + 文本 cosine 相似度） ──
        img_sims   = calculate_cosine_similarity(q_img, img_matrix)
        txt_sims   = calculate_cosine_similarity(q_txt, txt_matrix)
        coarse_scores = (alpha * img_sims + beta * txt_sims) / (alpha + beta)

        coarse_indices = np.argsort(coarse_scores)[::-1]
        candidate_uids   = []
        candidate_coarse = []

        for idx in coarse_indices:
            cand_uid = uid_array[idx]
            if cand_uid == q_uid:
                continue
            candidate_uids.append(cand_uid)
            candidate_coarse.append(coarse_scores[idx])
            if len(candidate_uids) >= top_n_coarse:
                break

        # ── 步骤 2：精排（融合粗筛得分 + 符号化结构匹配） ──
        final_scores = []
        for i, c_uid in enumerate(candidate_uids):
            c_struct = struct_data.get(str(c_uid), [])
            s_score  = symbolic_struct_sim(q_struct, c_struct)

            fused = (alpha + beta) * candidate_coarse[i] + gamma * s_score
            final_scores.append(fused)

        fine_indices = np.argsort(final_scores)[::-1]
        top_k_uids   = [candidate_uids[idx] for idx in fine_indices[:top_k_fine]]

        final_sim_mapping[q_uid] = top_k_uids

    return final_sim_mapping


if __name__ == "__main__":
    # 加载图像和文本特征
    with open("uid_to_image_features.pkl", "rb") as f:
        uid_to_img = pickle.load(f)
    with open("uid_to_text_features.pkl", "rb") as f:
        uid_to_txt = pickle.load(f)

    # 加载原始结构化 JSON（符号化匹配，不再使用向量）
    with open(r"C:\Work\pricai\gt_findings_structure_gpt54_dict.json", "r") as f:
        uid_to_struct = json.load(f)

    results = hierarchical_multimodal_retrieval(
        uid_to_img, uid_to_txt, uid_to_struct,
        alpha=0, beta=0, gamma=1,
        top_n_coarse=100, top_k_fine=3
    )

    with open("final_sim_mapping_top3_symbolic_0010.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ 符号化三模态检索完成！")