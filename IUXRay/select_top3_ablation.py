import numpy as np
import pickle
import json
from tqdm import tqdm


def perform_ablation_retrieval(img_features, txt_features, alpha=0.5, beta=0.5, top_k=3):
    """
    消融实验基准线：仅使用图像和文本特征进行检索
    S = alpha * Sim_img + beta * Sim_txt
    """
    uids = list(img_features.keys())
    uid_list = np.array(uids)

    # 将特征堆叠为矩阵以利用 NumPy 的矩阵运算加速
    # 假设特征在提取时已经过 L2 归一化
    img_matrix = np.array([img_features[uid] for uid in uids])
    txt_matrix = np.array([txt_features[uid] for uid in uids])

    # 如果没归一化，取消下面两行的注释（为了保险，建议执行）
    # img_matrix /= np.linalg.norm(img_matrix, axis=1, keepdims=True)
    # txt_matrix /= np.linalg.norm(txt_matrix, axis=1, keepdims=True)

    ablation_results = {}

    print(f"开始消融实验检索 (Alpha={alpha}, Beta={beta})...")

    # 使用矩阵乘法一次性计算全库相似度，速度极快
    for i, q_uid in enumerate(tqdm(uids)):
        q_img = img_features[q_uid]
        q_txt = txt_features[q_uid]

        # 计算余弦相似度向量 (1 x N)
        img_sims = np.dot(img_matrix, q_img)
        txt_sims = np.dot(txt_matrix, q_txt)

        # 融合得分
        combined_scores = alpha * img_sims + beta * txt_sims

        # 排序：argsort 返回从小到大的索引，因此取最后几个并翻转
        # 为了剔除自己，我们取 Top K+1
        sorted_indices = np.argsort(combined_scores)[::-1]

        current_top_k = []
        for idx in sorted_indices:
            cand_uid = uid_list[idx]
            if cand_uid == q_uid:  # 排除自己
                continue
            current_top_k.append(cand_uid)
            if len(current_top_k) >= top_k:
                break

        ablation_results[q_uid] = current_top_k

    return ablation_results


if __name__ == "__main__":
    # 加载已有的特征文件
    print("加载特征中...")
    with open("uid_to_image_features.pkl", "rb") as f:
        uid_to_img = pickle.load(f)
    with open("uid_to_text_features.pkl", "rb") as f:
        uid_to_txt = pickle.load(f)

    # 运行消融实验检索
    # 建议 alpha 和 beta 各占 0.5 作为一个强力的 Vanilla RAG Baseline
    sim_mapping_baseline = perform_ablation_retrieval(
        uid_to_img, uid_to_txt, alpha=0.5, beta=0.5, top_k=3
    )

    # 保存消融实验结果
    save_path = "ablation_sim_mapping_top3_img_text.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sim_mapping_baseline, f, indent=4)

    print(f"✅ 消融实验基准字典已保存至: {save_path}")