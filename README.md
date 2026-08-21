# Structure-Guided RAG with Counterfactual Exemplar Synthesis for Radiology Impression Generation

This repository contains the official implementation of our paper published at **PRICAI 2026**.

Experiments are provided for three chest X-ray datasets:

- IU X-Ray (`IUXRay/`)
- MIMIC-CXR (`mimic-cxr/`)
- JMID (`jmid/`, Japanese reports)

## Setup

Python 3.10+ and a CUDA-enabled GPU are recommended.

```bash
conda create -n radiology-rag python=3.10 -y
conda activate radiology-rag

pip install torch torchvision transformers datasets pillow tqdm numpy scipy
pip install open-clip-torch huggingface-hub openai qwen-vl-utils
pip install bert-score f1-radgraph nlgeval rank-bm25 FlagEmbedding
```

> The scripts contain machine-specific dataset, model, checkpoint, and output paths. Update these paths before running the code. API-based scripts also require valid credentials.

## Workflow

The following commands use MIMIC-CXR as an example. Equivalent scripts are available under `IUXRay/` and `jmid/`.

### 1. Prepare the test set

Update `DATASET_NAME` in `prepare_mimic_test.py`, then run:

```bash
python mimic-cxr/prepare_mimic_test.py
```

The script randomly selects up to 3,000 valid samples using seed `42`.

### 2. Generate structured findings

```bash
python mimic-cxr/generate_structure_findings.py
```

This converts free-text findings into structured JSON containing category, anatomy, observation, status, and attributes.

### 3. Extract features

```bash
python mimic-cxr/extract_findings_structure_feature.py
python mimic-cxr/extract_img_feature.py
python mimic-cxr/extract_text_feature.py
```

Configure the local BiomedCLIP checkpoint paths before feature extraction.

### 4. Retrieve the top three examples

```bash
python mimic-cxr/select_top3_jaccard_counterfactual_dual.py
```

The default similarity weights are:

```text
alpha = 0.4   # image
beta  = 0.4   # text
gamma = 0.2   # structured findings
```

Other retrieval baselines include BM25, MedCPT, MedRAG, and RadGraph.

### 5. Run report generation

Baseline methods and the proposed method are located in each dataset's `method/` directory. For example:

```bash
python mimic-cxr/method/qwen3vl8b_baseline.py
python mimic-cxr/method/qwen3vl8b_ours_counterfactual.py
```

Update the model, data, image, retrieval-mapping, and output paths in the selected script before running it.

## Ablation Studies

- `select_top3_ablation.py`: image and text retrieval only.
- `select_top3_jaccard_nocounterfactual.py`: structured retrieval without counterfactual editing.
- `select_top3_jaccard_counterfactual_dual.py`: full method; edit `alpha`, `beta`, and `gamma` to test different weights.

## Evaluation

Evaluate one result file after updating its path:

```bash
python mimic-cxr/evaluation_bleu.py
python mimic-cxr/evaluation_entity.py
```

For batch evaluation, update `folder_path` and run:

```bash
python evaluation_bleu_batch.py
python evaluation_entity_batch.py
python evaluation_bertscore_batch.py
```

For JMID, use the corresponding `*_jp.py` scripts. `IUXRay/evaluation_llm.py` provides multimodal LLM-as-a-judge evaluation; only the Gemini judge result was used in the final paper.

## Notes

- Run the scripts in the order shown above.
- The datasets and model checkpoints are not included.
- Filenames such as `442` represent `alpha=0.4`, `beta=0.4`, and `gamma=0.2`.
- This code is intended for research only, not for clinical use.
