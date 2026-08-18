import json
import os
import torch
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


def load_biomedclip_local(device):
    """
    使用你提供的逻辑，从本地检查点加载 BiomedCLIP
    """
    # print("正在检查/下载 BiomedCLIP 模型权重和配置文件...")
    # os.makedirs("checkpoints", exist_ok=True)
    #
    # hf_hub_download(
    #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     filename="open_clip_pytorch_model.bin",
    #     local_dir="checkpoints"
    # )
    # hf_hub_download(
    #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     filename="open_clip_config.json",
    #     local_dir="checkpoints"
    # )

    model_name = "biomedclip_local"
    with open(".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    print("正在初始化模型...")
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )

    model = model.to(device)
    model.eval()
    return model, preprocess


def extract_frontal_image_embeddings(json_path, image_base_dir, batch_size=32):
    """
    读取 JSON 文件，提取 Frontal 图片特征，返回 {uid: embedding} 字典
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"当前使用的计算设备: {device}")

    # 加载模型和预处理函数
    model, preprocess = load_biomedclip_local(device)

    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功加载 JSON 文件，共 {len(data)} 条记录。")

    uid_to_embedding = {}

    # 批量处理 (Batching) 提升显卡利用率
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="提取图片特征"):
            batch_data = data[i: i + batch_size]

            valid_images = []
            valid_uids = []

            for item in batch_data:
                uid = str(item.get('uid', 'unknown'))

                # 使用 .get() 防止键不存在报错，并且处理值为 null 的情况
                img_name = item.get('Frontal')

                # 【新增防御代码】：如果 img_name 是 None 或者空字符串，直接跳过
                if img_name is None or str(img_name).strip() == "":
                    print(f"\n跳过 UID: {uid}，因为 JSON 中缺少 'Frontal' 图片文件名。")
                    continue

                # 拼接完整的图片路径
                img_path = os.path.join(image_base_dir, str(img_name))
                try:
                    # 读取图片并预处理
                    img = Image.open(img_path).convert('RGB')
                    processed_img = preprocess(img)
                    valid_images.append(processed_img)
                    valid_uids.append(uid)
                except Exception as e:
                    print(f"\n警告：读取或处理图片 {img_path} 失败 (UID: {uid}) - 错误: {e}")

            # 如果这个 batch 里面有成功加载的图片
            if len(valid_images) > 0:
                # 堆叠成 tensor 并送入 GPU/CPU
                images_tensor = torch.stack(valid_images).to(device)

                # 仅调用图像编码器塔，提取特征
                image_features = model.encode_image(images_tensor)
                #print('image_features')
                #print(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # 转移到 CPU，转换为 float32 的 numpy 数组
                image_features_np = image_features.cpu().numpy().astype(np.float32)

                # 将结果写入字典
                for idx, uid in enumerate(valid_uids):
                    uid_to_embedding[uid] = image_features_np[idx]

    return uid_to_embedding


if __name__ == "__main__":
    # ==================== 配置路径 ====================
    # 替换为你实际的 JSON 文件路径
    JSON_FILE_PATH = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"

    # 替换为存放 Frontal 图片的文件夹路径
    IMAGE_DIRECTORY = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_images"

    # 如果你的显卡显存较大 (比如 16GB/24GB)，可以将 batch_size 调大到 64 或 128
    BATCH_SIZE = 32
    # ==================================================

    # 1. 运行提取函数
    result_dict = extract_frontal_image_embeddings(JSON_FILE_PATH, IMAGE_DIRECTORY, BATCH_SIZE)

    print(f"\n✅ 提取完成！成功提取了 {len(result_dict)} 个样本的图像特征。")

    # 2. 保存字典到文件，方便后续检索程序读取
    save_path = "mimic_uid_to_image_features.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"已将字典保存至: {save_path}")

    # ================= 补充说明：如何加载这个字典 =================
    # 在你的检索代码中，你可以通过以下方式直接加载并使用：
    import pickle
    with open("mimic_uid_to_image_features.pkl", "rb") as f:
        uid_to_img_emb = pickle.load(f)
    print("示例 UID '1' 的向量形状:", uid_to_img_emb["1"].shape)
    print("示例 UID '1' 的向量:", uid_to_img_emb["1"])