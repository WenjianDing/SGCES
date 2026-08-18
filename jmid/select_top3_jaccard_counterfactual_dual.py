import json
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def calculate_cosine_similarity(vec_q, matrix_db):
    """Cosine similarity implemented as dot product over L2-normalized vectors."""
    return np.dot(matrix_db, vec_q)


def token_overlap(a, b):
    """Token-level Jaccard similarity with simple null handling."""
    a_empty = (not a or str(a).strip().lower() in ("none", "null", ""))
    b_empty = (not b or str(b).strip().lower() in ("none", "null", ""))
    if a_empty and b_empty:
        return 1.0
    if a_empty or b_empty:
        return 0.5

    a_tokens = set(str(a).lower().split())
    b_tokens = set(str(b).lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def safe_token_overlap(str1, str2):
    """Safely handle None values before token overlap."""
    if not str1 and not str2:
        return 1.0
    if bool(str1) != bool(str2):
        return 0.0
    return token_overlap(str1, str2)


def match_single_obs(q_obs, c_obs):
    """Hierarchical matching score for one structured observation pair."""
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
    Return structural similarity plus matching indices/matrix for counterfactual editing.
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


def build_bidirectional_counterfactual_example(
    q_obs_list,
    c_obs_list,
    row_ind,
    col_ind,
    score_matrix,
    match_threshold=0.1,
):
    """
    Bidirectional counterfactual stitching:
    1. Prune redundant positive candidate findings.
    2. Preserve/normalize matched findings.
    3. Inject unmatched positive query findings.
    """
    c_matched_indices = set()
    q_matched_indices = set()

    for q_idx, c_idx in zip(row_ind, col_ind):
        if score_matrix[q_idx, c_idx] >= match_threshold:
            c_matched_indices.add(c_idx)
            q_matched_indices.add(q_idx)

    edited_c_obs_list = []

    for j, c_obs in enumerate(c_obs_list):
        if j in c_matched_indices:
            q_idx = list(row_ind)[list(col_ind).index(j)]
            q_obs = q_obs_list[q_idx]

            edited_obs = c_obs.copy()
            if str(edited_obs.get("Status", "")).lower() != str(q_obs.get("Status", "")).lower():
                edited_obs["Status"] = q_obs.get("Status")

            edited_c_obs_list.append(edited_obs)
        else:
            status = str(c_obs.get("Status", "")).lower()
            if status not in ["present", "abnormal"]:
                edited_c_obs_list.append(c_obs)

    for i, q_obs in enumerate(q_obs_list):
        if i not in q_matched_indices:
            status = str(q_obs.get("Status", "")).lower()
            if status in ["present", "abnormal"]:
                edited_c_obs_list.append(q_obs.copy())

    return edited_c_obs_list


def hierarchical_text_structure_retrieval(
    txt_features,
    struct_data,
    text_weight=0.6,
    struct_weight=0.4,
    top_n_coarse=100,
    top_k_fine=3,
):
    """
    Text-only coarse retrieval plus structural re-ranking.
    Final score = text_weight * text_similarity + struct_weight * structural_score.
    """
    uids = list(txt_features.keys())
    uid_array = np.array(uids)

    txt_matrix = np.array([txt_features[uid] for uid in uids])

    final_sim_mapping = {}
    final_edited_mapping = {}

    for q_uid in tqdm(uids, desc="Text-structure retrieval and counterfactual editing"):
        q_txt = txt_features[q_uid]
        q_struct = struct_data.get(str(q_uid), [])

        text_scores = calculate_cosine_similarity(q_txt, txt_matrix)
        coarse_indices = np.argsort(text_scores)[::-1]

        candidate_uids = []
        candidate_text_scores = []
        for idx in coarse_indices:
            cand_uid = uid_array[idx]
            if cand_uid == q_uid:
                continue
            candidate_uids.append(cand_uid)
            candidate_text_scores.append(text_scores[idx])
            if len(candidate_uids) >= top_n_coarse:
                break

        final_scores = []
        c_diff_info = []

        for i, c_uid in enumerate(candidate_uids):
            c_struct = struct_data.get(str(c_uid), [])
            s_score, row_ind, col_ind, score_matrix = symbolic_struct_sim(q_struct, c_struct)

            fused = text_weight * candidate_text_scores[i] + struct_weight * s_score
            final_scores.append(fused)
            c_diff_info.append((c_struct, row_ind, col_ind, score_matrix))

        fine_indices = np.argsort(final_scores)[::-1]

        top_k_uids = []
        top_k_edited_structs = []

        for idx in fine_indices[:top_k_fine]:
            c_uid = candidate_uids[idx]
            top_k_uids.append(c_uid)

            c_struct, row_ind, col_ind, score_matrix = c_diff_info[idx]
            edited_struct = build_bidirectional_counterfactual_example(
                q_struct, c_struct, row_ind, col_ind, score_matrix
            )
            top_k_edited_structs.append(edited_struct)

        final_sim_mapping[q_uid] = top_k_uids
        final_edited_mapping[q_uid] = top_k_edited_structs

    return final_sim_mapping, final_edited_mapping


if __name__ == "__main__":
    with open(r"C:\Work\pricai\jmid\jmid_uid_to_text_features_bgem3.pkl", "rb") as f:
        uid_to_txt = pickle.load(f)

    with open(r"C:\Work\pricai\jmid\jmid_test\jmid_findings_structure_gpt54_dict.json", "r", encoding="utf-8") as f:
        uid_to_struct = json.load(f)

    sim_results, edited_results = hierarchical_text_structure_retrieval(
        uid_to_txt,
        uid_to_struct,
        text_weight=0.6,
        struct_weight=0.4,
        top_n_coarse=100,
        top_k_fine=3,
    )

    with open("jmid_sim_mapping_top3_symbolic_64.json", "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=4, ensure_ascii=False)

    with open("jmid_dual_counterfactual_top3_64.json", "w", encoding="utf-8") as f:
        json.dump(edited_results, f, indent=4, ensure_ascii=False)

    print("Done. Generated text-structure retrieval and edited exemplar JSON files.")
