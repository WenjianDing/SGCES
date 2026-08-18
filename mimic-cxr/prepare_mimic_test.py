# coding=utf-8
"""
prepare_mimic_test.py
=====================
从 HuggingFace 加载 mimic-cxr-dataset，
随机抽取 3000 条保存为测试集。

图片保存为 PNG 文件，文本数据保存为 JSON。

输出：
  mimic_test/
    images/          所有测试图片 (PNG)
    mimic_test.json  测试数据列表，格式与 ct_data.json 一致：
                     [{uid, findings, impression, Frontal}, ...]

依赖安装：
  pip install datasets pillow tqdm
"""

import json
import os
import random
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

# ══════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════
DATASET_NAME  = r"C:\Users\81163\Downloads\mimic-cxr-dataset"
OUTPUT_DIR    = "mimic_test"
IMAGE_DIR     = os.path.join(OUTPUT_DIR, "mimic_test_images")
JSON_PATH     = os.path.join(OUTPUT_DIR, "mimic_test_data.json")
NUM_TEST      = 3000
RANDOM_SEED   = 42
# ══════════════════════════════════════════════════════════


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 1. 加载数据集
    print(f"Loading dataset: {DATASET_NAME} ...")
    print("（首次运行会下载数据，请耐心等待）")
    dataset = load_dataset(DATASET_NAME, split="train")
    total = len(dataset)
    print(f"  Total samples: {total}")

    # 2. 随机抽取 3000 条
    random.seed(RANDOM_SEED)
    indices = random.sample(range(total), min(NUM_TEST, total))
    indices.sort()
    print(f"  Sampled {len(indices)} samples (seed={RANDOM_SEED})")

    # 调试：打印第一条的 image 实际类型
    first_item = dataset[indices[0]]
    img_data_debug = first_item["image"]
    print(f"  DEBUG image type: {type(img_data_debug)}, value preview: {repr(img_data_debug)[:80]}")

    # 3. 逐条处理：保存图片 + 构造 JSON 记录
    print(f"\nSaving images to {IMAGE_DIR} ...")
    records = []
    skipped = 0

    for uid_idx, orig_idx in enumerate(tqdm(indices, desc="Processing")):
        item = dataset[orig_idx]

        uid       = str(uid_idx + 1)           # 从 1 开始的连续 uid
        findings  = item.get("findings") or ""
        impression = item.get("impression") or ""

        # 跳过 findings 或 impression 为空的样本
        if not findings or not findings.strip() or not impression or not impression.strip():
            skipped += 1
            continue

        # 保存图片（image 字段是 binary 原始字节）
        img_filename = f"{uid}.png"
        img_path     = os.path.join(IMAGE_DIR, img_filename)

        try:
            import io
            img_data = item["image"]
            # 统一用 BytesIO 尝试，适配 binary/bytes/memoryview 等所有二进制格式
            if isinstance(img_data, (bytes, bytearray, memoryview)):
                img = Image.open(io.BytesIO(bytes(img_data))).convert("RGB")
            elif isinstance(img_data, Image.Image):
                img = img_data.convert("RGB")
            else:
                # 最后尝试：强转 bytes
                img = Image.open(io.BytesIO(bytes(img_data))).convert("RGB")
            img.save(img_path)
        except Exception as e:
            print(f"WARNING: Failed to save image for uid={uid}: {e}")
            skipped += 1
            continue

        records.append({
            "uid":        uid,
            "findings":   findings.strip(),
            "impression": impression.strip(),
            "Frontal":    img_filename,
        })

    # 4. 保存 JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完成！")
    print(f"   有效样本: {len(records)}")
    print(f"   跳过样本: {skipped}（findings 或 impression 为空）")
    print(f"   图片目录: {IMAGE_DIR}")
    print(f"   JSON 文件: {JSON_PATH}")

    # 5. 打印示例
    print("\n示例记录（前2条）：")
    for r in records[:2]:
        print(f"  uid={r['uid']}")
        print(f"    findings:   {r['findings'][:60]}...")
        print(f"    impression: {r['impression'][:60]}...")
        print(f"    image:      {r['Frontal']}")


if __name__ == "__main__":
    main()
