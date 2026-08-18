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
    if not str1 and not str2:
        return 1.0
    if bool(str1) != bool(str2):
        return 0.0
    return token_overlap(str1, str2)


def match_single_obs(q_obs, c_obs):
    """单个 observation 之间的层次化匹配分数"""
    if q_obs.get("Category") != c_obs.get("Category"):
        return 0.0
    if q_obs.get("Status") != c_obs.get("Status"):
        return 0.0

    obs_score = token_overlap(q_obs.get("Observation"), c_obs.get("Observation"))
    anat_score = token_overlap(q_obs.get("Anatomy"), c_obs.get("Anatomy"))
    attr_score = token_overlap(q_obs.get("Attributes"), c_obs.get("Attributes"))

    return 0.5 * obs_score + 0.3 * anat_score + 0.2 * attr_score


def symbolic_struct_sim(q_obs_list, c_obs_list):
    """
    计算基于匈牙利算法的结构化相似度分数。
    由于这是 Baseline，我们只需要分数用于检索排序，不需要返回匹配矩阵。
    """
    n = len(q_obs_list)
    m = len(c_obs_list)

    if n == 0 or m == 0:
        return 0.0

    score_matrix = np.zeros((n, m))
    for i, q_obs in enumerate(q_obs_list):
        for j, c_obs in enumerate(c_obs_list):
            score_matrix[i, j] = match_single_obs(q_obs, c_obs)

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    matched_scores = score_matrix[row_ind, col_ind]
    score = float(matched_scores.sum() / n) if n > 0 else 0.0

    return score


def hierarchical_multimodal_retrieval(img_features, txt_features, struct_data,
                                      alpha=0.2, beta=0.2, gamma=0.6,
                                      top_n_coarse=50, top_k_fine=3):
    uids = list(img_features.keys())
    uid_array = np.array(uids)

    img_matrix = np.array([img_features[uid] for uid in uids])
    txt_matrix = np.array([txt_features[uid] for uid in uids])

    final_sim_mapping = {}
    final_raw_mapping = {}  # 用于存储未编辑的原始结构化样本列表 (Baseline)

    for q_uid in tqdm(uids, desc="三模态符号化检索 (Baseline) 中"):
        q_img = img_features[q_uid]
        q_txt = txt_features[q_uid]
        q_struct = struct_data.get(str(q_uid), [])

        # ── 步骤 1：粗筛 ──
        img_sims = calculate_cosine_similarity(q_img, img_matrix)
        txt_sims = calculate_cosine_similarity(q_txt, txt_matrix)
        coarse_scores = (alpha * img_sims + beta * txt_sims) / (alpha + beta)

        coarse_indices = np.argsort(coarse_scores)[::-1]
        candidate_uids = []
        candidate_coarse = []

        for idx in coarse_indices:
            cand_uid = uid_array[idx]
            if cand_uid == q_uid:
                continue
            candidate_uids.append(cand_uid)
            candidate_coarse.append(coarse_scores[idx])
            if len(candidate_uids) >= top_n_coarse:
                break

        # ── 步骤 2：精排 ──
        final_scores = []
        for i, c_uid in enumerate(candidate_uids):
            c_struct = struct_data.get(str(c_uid), [])
            # Baseline 中，只需获取相似度分数用于排序
            s_score = symbolic_struct_sim(q_struct, c_struct)

            fused = (alpha + beta) * candidate_coarse[i] + gamma * s_score
            final_scores.append(fused)

        fine_indices = np.argsort(final_scores)[::-1]

        top_k_uids = []
        top_k_raw_structs = []

        # ── 步骤 3：直接提取原始结构化信息 (不做反事实编辑) ──
        for idx in fine_indices[:top_k_fine]:
            c_uid = candidate_uids[idx]
            top_k_uids.append(c_uid)

            # 获取参考样本最原始的结构化 JSON
            c_struct_raw = struct_data.get(str(c_uid), [])
            top_k_raw_structs.append(c_struct_raw)

        final_sim_mapping[q_uid] = top_k_uids
        final_raw_mapping[q_uid] = top_k_raw_structs

    return final_sim_mapping, final_raw_mapping


if __name__ == "__main__":
    # 加载图像和文本特征
    with open(r"C:\Work\pricai\IUXRay\generated_data\uid_to_image_features.pkl", "rb") as f:
        uid_to_img = pickle.load(f)
    with open(r"C:\Work\pricai\IUXRay\generated_data\uid_to_text_features.pkl", "rb") as f:
        uid_to_txt = pickle.load(f)

    # 加载原始结构化 JSON
    with open(r"C:\Work\pricai\IUXRay\generated_data\gt_findings_structure_gpt54_dict.json", "r", encoding="utf-8") as f:
        uid_to_struct = json.load(f)

    sim_results, raw_results = hierarchical_multimodal_retrieval(
        uid_to_img, uid_to_txt, uid_to_struct,
        alpha=0.4, beta=0.4, gamma=0.2,
        top_n_coarse=100, top_k_fine=3
    )

    # 1. 保存 Top-3 相似度 UID 映射 (名字保持一致，证明检索基础没变)
    with open("final_sim_mapping_top3_symbolic_442.json", "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=4, ensure_ascii=False)

    # 2. 保存未修改的原始结构化样本，用作 Baseline 评测
    with open("final_baseline_raw_struct_top3.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=4, ensure_ascii=False)

    print("✅ Baseline：符号化三模态检索完成！已生成包含原始参考信息的 final_baseline_raw_struct_top3.json 文件。")