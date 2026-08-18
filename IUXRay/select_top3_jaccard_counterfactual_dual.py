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
    修改点 1：除了返回相似度，同时返回用于反事实编辑的矩阵和索引
    """
    n = len(q_obs_list)
    m = len(c_obs_list)

    if n == 0 or m == 0:
        return 0.0, [], [], np.zeros((0, 0))

    score_matrix = np.zeros((n, m))
    for i, q_obs in enumerate(q_obs_list):
        for j, c_obs in enumerate(c_obs_list):
            score_matrix[i, j] = match_single_obs(q_obs, c_obs)

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    matched_scores = score_matrix[row_ind, col_ind]
    score = float(matched_scores.sum() / n) if n > 0 else 0.0

    return score, row_ind, col_ind, score_matrix


def build_bidirectional_counterfactual_example(q_obs_list, c_obs_list, row_ind, col_ind, score_matrix,
                                               match_threshold=0.1):
    """
    升级版：双向反事实缝合 (Bi-directional Counterfactual Stitching)
    包含：1. 冗余剪裁 (Pruning)  2. 状态覆写 (Overwrite)  3. 缺失注入 (Injection)
    """
    c_matched_indices = set()
    q_matched_indices = set()

    for q_idx, c_idx in zip(row_ind, col_ind):
        # 只有相似度过阈值，才算真正的匹配
        if score_matrix[q_idx, c_idx] >= match_threshold:
            c_matched_indices.add(c_idx)
            q_matched_indices.add(q_idx)

    edited_c_obs_list = []

    # --- 阶段 1：处理 Candidate (剪裁与覆写) ---
    for j, c_obs in enumerate(c_obs_list):
        if j in c_matched_indices:
            # 找到与其匹配的 Query 节点
            q_idx = list(row_ind)[list(col_ind).index(j)]
            q_obs = q_obs_list[q_idx]

            # 【升级点 1：状态覆写】深拷贝防止污染原始数据
            edited_obs = c_obs.copy()
            if str(edited_obs.get("Status", "")).lower() != str(q_obs.get("Status", "")).lower():

                edited_obs["Status"] = q_obs.get("Status")  # 强行对齐为当前病人的状态

            edited_c_obs_list.append(edited_obs)
        else:
            # 【保留原有逻辑：剪裁冗余阳性】
            status = str(c_obs.get("Status", "")).lower()
            if status not in ["present", "abnormal"]:
                edited_c_obs_list.append(c_obs)


    #--- 阶段 2：处理 Query (特征注入防漏诊) ---
    for i, q_obs in enumerate(q_obs_list):
        if i not in q_matched_indices:
            # 【升级点 2：特征注入】如果当前病人有关键的阳性病灶，但参考样本没匹配上，强行塞进去
            status = str(q_obs.get("Status", "")).lower()
            if status in ["present", "suspected"]:

                edited_c_obs_list.append(q_obs.copy())

    return edited_c_obs_list


def hierarchical_multimodal_retrieval(img_features, txt_features, struct_data,
                                      alpha=0.2, beta=0.2, gamma=0.6,
                                      top_n_coarse=50, top_k_fine=3):
    uids = list(img_features.keys())
    uid_array = np.array(uids)

    img_matrix = np.array([img_features[uid] for uid in uids])
    txt_matrix = np.array([txt_features[uid] for uid in uids])

    final_sim_mapping = {}
    final_edited_mapping = {}  # 新增：用于存储编辑后的结构化样本列表

    for q_uid in tqdm(uids, desc="三模态符号化检索与反事实编辑中"):

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
        c_diff_info = []  # 缓存矩阵信息

        for i, c_uid in enumerate(candidate_uids):
            c_struct = struct_data.get(str(c_uid), [])
            # 接收返回的矩阵信息
            s_score, row_ind, col_ind, score_matrix = symbolic_struct_sim(q_struct, c_struct)

            fused = (alpha + beta) * candidate_coarse[i] + gamma * s_score
            final_scores.append(fused)
            c_diff_info.append((c_struct, row_ind, col_ind, score_matrix))

        fine_indices = np.argsort(final_scores)[::-1]

        top_k_uids = []
        top_k_edited_structs = []

        # ── 步骤 3：反事实编辑提取 ──
        for idx in fine_indices[:top_k_fine]:
            c_uid = candidate_uids[idx]
            top_k_uids.append(c_uid)
            #print('c_uid:', c_uid)
            c_struct, row_ind, col_ind, score_matrix = c_diff_info[idx]

            # 替换为升级后的双向缝合函数
            # 注意：这里需要额外把 q_struct 传进去，以便进行状态比对和注入
            edited_struct = build_bidirectional_counterfactual_example(
                q_struct, c_struct, row_ind, col_ind, score_matrix
            )

            top_k_edited_structs.append(edited_struct)

        final_sim_mapping[q_uid] = top_k_uids
        final_edited_mapping[q_uid] = top_k_edited_structs

    return final_sim_mapping, final_edited_mapping


if __name__ == "__main__":
    # 加载图像和文本特征
    with open(r"C:\Work\pricai\IUXRay\generated_data\uid_to_image_features.pkl", "rb") as f:
        uid_to_img = pickle.load(f)
    with open(r"C:\Work\pricai\IUXRay\generated_data\uid_to_text_features.pkl", "rb") as f:
        uid_to_txt = pickle.load(f)

    # 加载原始结构化 JSON
    with open(r"C:\Work\pricai\IUXRay\generated_data\gt_findings_structure_gpt54_dict.json", "r", encoding="utf-8") as f:
        uid_to_struct = json.load(f)
    #这里的alpha beta是论文中的beta gamma, 这里的gamma是论文中的alpha
    sim_results, edited_results = hierarchical_multimodal_retrieval(
        uid_to_img, uid_to_txt, uid_to_struct,
        alpha=0.4, beta=0.4, gamma=0.2,
        top_n_coarse=100, top_k_fine=3
    )

    # 1. 保存 Top-3 相似度 UID 映射
    with open("final_sim_mapping_top3_symbolic_442.json", "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=4, ensure_ascii=False)

    # 2. 保存反事实编辑后的 3 个结构化样本 (Golden Exemplars)
    with open("final_dual_counterfactual_edited_top3_442.json", "w", encoding="utf-8") as f:
        json.dump(edited_results, f, indent=4, ensure_ascii=False)

    print("✅ 符号化三模态检索及反事实编辑完成！已生成纯净的 Edited JSON 文件。")