import json
import random
from collections import Counter
import re

test_data_path = r"C:\Work\pricai\ct_data.json"
save_data_path = r"C:\Work\pricai\ct_few_shot_examples.json"

# 固定随机种子，保证实验可严格复现
random.seed(42)

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# ==========================================
# 步骤 1: 统计提取 Top 10 的疾病类别
# ==========================================
all_atomic_problems = []
for data in test_data:
    problems_str = data.get('Problems', '')
    if problems_str:
        # 使用逗号或分号拆分，提取基础标签（根据你的数据格式可微调正则）
        atomic_list = [p.strip().lower() for p in re.split(r'[,;]', problems_str) if p.strip()]
        all_atomic_problems.extend(atomic_list)

problem_counts = Counter(all_atomic_problems)
top_10_keys = [item[0] for item in problem_counts.most_common(10)]
print(f"Top 10 类别将被作为分桶基准: {top_10_keys}\n")

# ==========================================
# 步骤 2: 将所有样本分发到对应的 10 个（+1个兜底）类别桶中
# ==========================================
category2samples = {key: [] for key in top_10_keys}
category2samples["other_complex"] = []  # 兜底类别
uid2category = {}  # 记录每个 UID 被分配到了哪个桶

for data in test_data:
    each_uid = data.get('uid')
    problems_str = data.get('Problems', '')

    atomic_list = [p.strip().lower() for p in re.split(r'[,;]', problems_str) if p.strip()] if problems_str else []

    assigned_category = "other_complex"

    # 分配逻辑
    if "normal" in atomic_list and "normal" in category2samples:
        assigned_category = "normal"
    else:
        for p in atomic_list:
            if p in top_10_keys:
                assigned_category = p
                break

    category2samples[assigned_category].append(data)
    uid2category[each_uid] = assigned_category

# ==========================================
# 步骤 3: 针对每个 UID，在其所属类别桶内随机抽取 3 个 Few-shot
# ==========================================
uid2fewshots = {}

for data in test_data:
    target_uid = data.get('uid')
    target_category = uid2category[target_uid]

    # 核心：在“同类别桶”内构建候选池，且必须排除当前测试样本自身！
    candidate_pool = [
        sample for sample in category2samples[target_category]
        if sample.get('uid') != target_uid
    ]

    # 防止某些罕见类别（如 other_complex）候选数量不足 3 个的情况
    sample_size = min(3, len(candidate_pool))

    # 无放回随机抽样
    sampled_examples = random.sample(candidate_pool, sample_size)

    # 提取需要的字段并保存
    few_shots_list = []
    for ex in sampled_examples:
        few_shots_list.append({
            "uid": ex.get('uid'),
            "category": target_category,  # 记录一下它是从哪个类别池抽出来的，方便你 debug
            "findings": ex.get('findings', ''),
            "impression": ex.get('impression', '')
        })

    uid2fewshots[target_uid] = few_shots_list

# ==========================================
# 步骤 4: 保存为 JSON 字典
# ==========================================
with open(save_data_path, 'w', encoding='utf-8') as f_out:
    json.dump(uid2fewshots, f_out, indent=4, ensure_ascii=False)

print(f"处理完成！已为 {len(uid2fewshots)} 个样本完成了类内 3-shot 采样。")
print(f"数据已保存至: {save_data_path}")