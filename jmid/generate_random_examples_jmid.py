# coding=utf-8
import json
import random

test_data_path = r"C:\Work\pricai\jmid\jmid_test\jmid_test_data.json"
save_data_path = r"C:\Work\pricai\jmid\jmid_test\jmid_few_shot_examples.json"

random.seed(42)

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

uid2fewshots = {}

for data in test_data:
    target_uid = data.get('uid')

    # 全局候选池，排除自身
    candidate_pool = [s for s in test_data if s.get('uid') != target_uid]

    sampled_examples = random.sample(candidate_pool, min(3, len(candidate_pool)))

    uid2fewshots[target_uid] = [
        {
            "uid":        ex.get('uid'),
            "findings":   ex.get('findings', ''),
            "impression": ex.get('impression', '')
        }
        for ex in sampled_examples
    ]

with open(save_data_path, 'w', encoding='utf-8') as f:
    json.dump(uid2fewshots, f, indent=4, ensure_ascii=False)

print(f"完成！已为 {len(uid2fewshots)} 个样本随机抽取 3 个 few-shot 示例。")
print(f"保存至: {save_data_path}")