# file: chemberta_embedding_extractor.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import warnings

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def load_model_and_tokenizer(model_dir: str, device: torch.device, local_only: bool = True):
    """科研级模型加载 - 支持快速分词器和完全离线模式"""
    try:
        tok = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,  # 启用快速分词器
            local_files_only=local_only,
            trust_remote_code=False  # 安全模式
        )
        mdl = AutoModel.from_pretrained(
            model_dir,
            local_files_only=local_only,
            trust_remote_code=False
        ).to(device).eval()
        return tok, mdl
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise


def canonicalize_smiles(smi: str, drop_stereo: bool = False):
    """
    科研级SMILES规范化
    遵循RDKit最佳实践，保留立体化学信息（默认）
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # 保留立体化学信息（除非明确要求去除）
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not drop_stereo)
    except Exception as e:
        print(f"Error canonicalizing {smi}: {str(e)}")
        return None


def load_smiles_file(input_path):
    """科研级SMILES加载 - 支持多种格式并过滤无效行"""
    if input_path.lower().endswith('.smi'):
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f]
        # 跳过空行和注释行
        return [l for l in lines if l and not l.startswith('#')]
    elif input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
        assert 'SMILES' in df.columns, "Excel文件必须包含SMILES列"
        return df['SMILES'].astype(str).tolist()
    else:  # CSV/TSV
        try:
            df = pd.read_csv(input_path, sep=None, engine='python', on_bad_lines='warn')
            assert 'SMILES' in df.columns, "CSV文件必须包含SMILES列"
            return df['SMILES'].astype(str).tolist()
        except Exception as e:
            raise ValueError(f"文件解析失败: {str(e)}")


@torch.inference_mode()
def extract_embeddings(smiles_list, tokenizer, model, device,
                       batch_size=256, max_length=256,
                       l2norm=False):
    """
    修改后的嵌入提取函数 - 仅提取最后一层的 [CLS] 标记嵌入
    符合高水平SCI论文标准
    """
    model.eval()

    # 获取模型配置信息
    hidden_size = model.config.hidden_size

    # 预分配内存
    num_samples = len(smiles_list)
    embeddings = np.zeros((num_samples, hidden_size), dtype=np.float32)
    valid_mask = np.zeros(num_samples, dtype=bool)

    # 进度条设置
    pbar = tqdm(total=num_samples, desc="提取嵌入", unit="smiles")

    i = 0
    while i < num_samples:
        batch_smiles = smiles_list[i:i + batch_size]
        current_batch_size = len(batch_smiles)

        try:
            # 批处理编码
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True  # 确保添加 [CLS] 和 [SEP] 标记
            ).to(device)

            # FP16加速 (如果可用)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(**inputs, output_hidden_states=True)

            # 科研级提取策略: 仅使用最后一层的 [CLS] 标记
            # 获取所有层的隐藏状态
            hidden_states = outputs.hidden_states

            # 提取最后一层的隐藏状态
            last_layer_hidden_states = hidden_states[-1]

            # 提取每个序列的 [CLS] 标记 (索引为0)
            cls_embeddings = last_layer_hidden_states[:, 0, :]

            # 转换为numpy数组
            batch_embeddings = cls_embeddings.cpu().numpy()

            # L2归一化 (可选)
            if l2norm:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-12)

            # 填充结果
            embeddings[i:i + current_batch_size] = batch_embeddings
            valid_mask[i:i + current_batch_size] = True

            # 成功处理，移动到下一批
            i += current_batch_size
            pbar.update(current_batch_size)

            # 清理显存 (不频繁执行)
            if device.type == 'cuda' and i % (10 * batch_size) == 0:
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # OOM处理：减小批大小并重试
            if batch_size > 1:
                new_batch_size = max(1, batch_size // 2)
                print(f"⚠️ OOM警告: 批大小从{batch_size}减小到{new_batch_size}")
                batch_size = new_batch_size
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise RuntimeError("批大小已减小到1但仍OOM，请检查模型或数据")

    pbar.close()
    return embeddings[valid_mask], [s for s, v in zip(smiles_list, valid_mask) if v]


def main():
    parser = argparse.ArgumentParser(
        description="科研级ChemBERTa嵌入提取工具\n符合高水平SCI论文标准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据参数
    parser.add_argument("--input", required=True,
                        help="输入文件(.smi/.csv/.xlsx)")
    parser.add_argument("--output", default="chemberta_embeddings",
                        help="输出文件前缀")

    # 模型参数
    parser.add_argument("--model", choices=["zinc"], default="zinc",
                        help="选择模型: zinc(seyonec) [脚本已修改为仅支持此选项]")
    parser.add_argument("--zinc_dir", required=True,
                        help="本地 seyonec/ChemBERTa-zinc-base-v1 目录")

    # 科学处理参数
    parser.add_argument("--batch_size", type=int, default=256,
                        help="初始批处理大小(自动调整)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="SMILES最大长度")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="计算设备")

    # 离线/联网控制
    parser.add_argument("--online", action="store_true",
                        help="启用在线模式(默认完全离线)")

    # 化学参数
    parser.add_argument("--drop_stereo", action="store_true",
                        help="去除立体化学(默认保留)")

    # 嵌入提取参数
    parser.add_argument("--l2norm", action="store_true",
                        help="L2归一化嵌入向量")

    args = parser.parse_args()

    # 科研级环境配置
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 设备配置
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print(f"🧪 科研级ChemBERTa嵌入提取工具")
    print("=" * 60)
    print(f"📁 输入文件: {args.input}")
    print(f"💾 输出前缀: {args.output}")
    print(f"⚙️  设备: {device} | 初始批大小: {args.batch_size}")
    print(f"🔬 提取策略: 最后一层 [CLS] 标记 | L2归一化={args.l2norm}")
    print(f"🧪 化学处理: 保留立体化学={not args.drop_stereo}")
    print(f"🌐 联网模式: {'启用' if args.online else '禁用'}")
    print("=" * 60)

    # 加载并规范化SMILES
    print("\n🔍 加载并预处理SMILES...")
    raw_smiles = load_smiles_file(args.input)
    print(f"  加载SMILES数量: {len(raw_smiles)}")

    canon_smiles = []
    invalid_indices = []
    for i, smi in enumerate(tqdm(raw_smiles, desc="规范化SMILES")):
        canon = canonicalize_smiles(smi, args.drop_stereo)
        if canon is None:
            invalid_indices.append(i)
        else:
            canon_smiles.append(canon)

    print(f"✅ 有效SMILES: {len(canon_smiles)} | ❌ 无效: {len(invalid_indices)}")

    # 处理无效SMILES
    if invalid_indices:
        invalid_df = pd.DataFrame({
            "original_index": invalid_indices,
            "original_smiles": [raw_smiles[i] for i in invalid_indices]
        })
        invalid_path = f"{args.output}_invalid_smiles.csv"
        invalid_df.to_csv(invalid_path, index=False)
        print(f"⚠️ 保存无效SMILES到: {invalid_path}")

    # ========= 模型处理 =========
    # 仅使用 seyonec/ChemBERTa-zinc-base-v1 模型
    tasks = [("zinc", args.zinc_dir)]

    # 显示警告如果用户尝试使用其他模型
    if args.model != "zinc":
        print(f"⚠️  警告: 脚本已修改为仅使用 'zinc' 模型，忽略 '{args.model}' 选项")

    for tag, model_dir in tasks:
        print("\n" + "=" * 60)
        print(f"🚀 处理模型: {tag.upper()} | 路径: {model_dir}")
        print("=" * 60)

        try:
            tokenizer, model = load_model_and_tokenizer(
                model_dir, device, local_only=not args.online
            )

            # 显示模型信息
            print(f"🔧 模型名称: {model.config._name_or_path}")
            print(f"🔧 隐藏层大小: {model.config.hidden_size}")
            print(f"🔧 总层数: {model.config.num_hidden_layers}")
            print(f"🔧 提取策略: 最后一层 [CLS] 标记")

            # 提取嵌入
            embeddings, valid_smiles = extract_embeddings(
                canon_smiles, tokenizer, model, device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                l2norm=args.l2norm
            )

            # 保存结果
            emb_path = f"{args.output}_{tag}.npy"
            meta_path = f"{args.output}_{tag}_metadata.csv"

            np.save(emb_path, embeddings)
            meta_df = pd.DataFrame({
                "original_index": [i for i, s in enumerate(raw_smiles)
                                   if canonicalize_smiles(s, args.drop_stereo) in valid_smiles],
                "original_smiles": [s for i, s in enumerate(raw_smiles)
                                   if canonicalize_smiles(s, args.drop_stereo) in valid_smiles],
                "canonical_smiles": valid_smiles
            })
            meta_df.to_csv(meta_path, index=False)

            print("\n✅ 完成!")
            print(f"  嵌入维度: {embeddings.shape}")
            print(f"  嵌入文件: {emb_path}")
            print(f"  元数据文件: {meta_path}")
            print(f"  嵌入统计: 均值={np.mean(embeddings):.4f} ± {np.std(embeddings):.4f}")

        except Exception as e:
            print(f"\n❌ 处理模型 {tag} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    main()