import json
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# 屏蔽 HuggingFace 的软链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_biomedclip_text_model(device):
    """
    加载 BiomedCLIP 模型和对应的 Tokenizer
    """

    model_name = "biomedclip_local"
    with open(".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    # 获取专用的 Tokenizer
    tokenizer = get_tokenizer(model_name)

    print("正在初始化文本编码器...")
    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained=".../BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin"
    )

    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_findings_embeddings(json_path, batch_size=64):
    """
    读取 JSON 文件，提取 Findings 文本特征，返回 {uid: embedding} 字典
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"当前使用的计算设备: {device}")

    model, tokenizer = load_biomedclip_text_model(device)

    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功加载 JSON 文件，共 {len(data)} 条记录。")

    uid_to_embedding = {}
    context_length = 256  # BiomedCLIP 固定的文本长度

    # 批量处理
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="提取文本特征"):
            batch_data = data[i: i + batch_size]

            valid_texts = []
            valid_uids = []

            for item in batch_data:
                uid = str(item.get('uid', 'unknown'))
                findings_text = item.get('findings')

                # 防御代码：处理 findings 为空的情况
                if findings_text is None or str(findings_text).strip() == "":
                    # 如果 Findings 为空，可以跳过或者填充一个默认字符串（如 "normal"）
                    # 这里建议跳过，因为没有 Findings 的数据对 RAG 检索没有意义
                    continue

                valid_texts.append(str(findings_text).strip())
                valid_uids.append(uid)

            if len(valid_texts) > 0:
                # 文本 Tokenize
                # 注意：BiomedCLIP 的 tokenizer 会自动处理截断 (Truncation)
                tokens = tokenizer(valid_texts, context_length=context_length).to(device)

                # 仅调用文本编码器塔 (Text Encoder)
                text_features = model.encode_text(tokens)
                # print('text_features')
                # print(text_features)
                # 归一化（通常 CLIP 检索需要 L2 归一化，这样点积就是余弦相似度）
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 转换为 numpy
                features_np = text_features.cpu().numpy().astype(np.float32)

                for idx, uid in enumerate(valid_uids):
                    uid_to_embedding[uid] = features_np[idx]

    return uid_to_embedding


if __name__ == "__main__":
    # ==================== 配置路径 ====================
    # 替换为你实际的 JSON 文件路径
    JSON_FILE_PATH = r"C:\Work\pricai\mimic-cxr\mimic_test\mimic_test_data.json"

    # 文本处理比图片快，batch_size 可以设置得稍大
    BATCH_SIZE = 64
    # ==================================================

    # 1. 执行提取
    text_result_dict = extract_findings_embeddings(JSON_FILE_PATH, BATCH_SIZE)

    print(f"\n✅ 提取完成！成功处理了 {len(text_result_dict)} 条 Findings 文本。")

    # 2. 保存结果
    save_path = "mimic_uid_to_text_features.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(text_result_dict, f)

    print(f"文本特征字典已保存至: {save_path}")


    import pickle
    with open("mimic_uid_to_text_features.pkl", "rb") as f:
        uid_to_text_emb = pickle.load(f)
    print("示例 UID '1' 的向量形状:", uid_to_text_emb["1"].shape)
    print("示例 UID '1' 的向量:", uid_to_text_emb["1"])

    # ================= 冲刺提示：多模态检索逻辑 =================
    # 现在你拥有了：
    # 1. uid_to_image_features.pkl (由图片脚本生成)
    # 2. uid_to_text_features.pkl (由本脚本生成)
    #
    # 当新的 Query 进来时：
    # score = alpha * cosine(query_img, db_img) + beta * cosine(query_txt, db_txt)
    # 因为我们在代码里加了 text_features /= norm，所以计算相似度只需用 np.dot 即可。