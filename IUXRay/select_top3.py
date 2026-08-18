import numpy as np
import pickle
import json
from tqdm import tqdm


def calculate_cosine_similarity(vec_q, matrix_db):
    """计算余弦相似度 (假设已 L2 归一化)"""
    return np.dot(matrix_db, vec_q)


def chamfer_similarity(query_struct_list, cand_struct_list):
    """计算结构化条目集合之间的倒角匹配得分"""
    # print('query_struct_list')
    # print(query_struct_list)
    # print('cand_struct_list')
    # print(cand_struct_list)
    if not query_struct_list or not cand_struct_list:
        return 0.0
    Q = np.array(query_struct_list)
    D = np.array(cand_struct_list)
    sim_matrix = np.dot(Q, D.T)
    return np.mean(np.max(sim_matrix, axis=1))


def hierarchical_multimodal_retrieval(img_features, txt_features, struct_features,
                                      alpha=0.3, beta=0.3, gamma=0.4,
                                      top_n_coarse=50, top_k_fine=3):
    """
    层次化三模态检索
    alpha: 图像权重, beta: 文本权重, gamma: 结构化权重
    约束条件: alpha + beta + gamma = 1
    """
    uids = list(img_features.keys())
    uid_list = np.array(uids)

    # 构建数据库矩阵以加速
    img_matrix = np.array([img_features[uid] for uid in uids])
    txt_matrix = np.array([txt_features[uid] for uid in uids])

    final_sim_mapping = {}

    for q_uid in tqdm(uids, desc="三模态对齐检索中"):
        q_img = img_features[q_uid]
        q_txt = txt_features[q_uid]
        q_struct = struct_features.get(q_uid, [])

        # --- 步骤 1: 粗筛 (考虑图像 + 文本) ---
        img_sims = calculate_cosine_similarity(q_img, img_matrix)
        txt_sims = calculate_cosine_similarity(q_txt, txt_matrix)

        # 归一化权重下的初步得分
        # 这里先用 alpha 和 beta 的相对比例进行筛选
        coarse_scores = (alpha * img_sims + beta * txt_sims) / (alpha + beta)

        # 排序并取 Top-N，排除自己
        coarse_indices = np.argsort(coarse_scores)[::-1]
        candidate_uids = []
        candidate_coarse_scores = []

        for idx in coarse_indices:
            cand_uid = uid_list[idx]
            if cand_uid == q_uid: continue
            candidate_uids.append(cand_uid)
            candidate_coarse_scores.append(coarse_scores[idx])
            if len(candidate_uids) >= top_n_coarse: break

        # --- 步骤 2: 精排 (融合 粗筛得分 + 结构化得分) ---
        final_scores_for_candidates = []
        for i, c_uid in enumerate(candidate_uids):
            c_struct = struct_features.get(c_uid, [])
            s_score = chamfer_similarity(q_struct, c_struct)

            # 【核心公式】：融合三个模态的信息
            # candidate_coarse_scores[i] 已经包含了 img 和 txt 的加权和
            # 我们按照 gamma 的配比，将结构化得分加入
            total_fused_score = (alpha + beta) * candidate_coarse_scores[i] + gamma * s_score
            final_scores_for_candidates.append(total_fused_score)

        # 根据最终融合总分排序
        fine_indices = np.argsort(final_scores_for_candidates)[::-1]
        top_k_uids = [candidate_uids[idx] for idx in fine_indices[:top_k_fine]]

        final_sim_mapping[q_uid] = top_k_uids

    return final_sim_mapping


if __name__ == "__main__":
    # 加载特征
    with open("uid_to_image_features.pkl", "rb") as f: uid_to_img = pickle.load(f)
    with open("uid_to_text_features.pkl", "rb") as f: uid_to_txt = pickle.load(f)
    with open("uid_to_structured_features_gpt54.pkl", "rb") as f: uid_to_struct = pickle.load(f)

    # 执行三模态融合检索
    # 在 10 天冲刺中，建议 gamma 设置在 0.4-0.6 之间，因为结构化对齐是你的主要创新点
    results = hierarchical_multimodal_retrieval(
        uid_to_img, uid_to_txt, uid_to_struct,
        alpha=0, beta=0.5, gamma=0.5,
        top_n_coarse=100, top_k_fine=3
    )

    with open("final_sim_mapping_top3_fused_055_gpt54.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ 三模态对齐检索完成！")