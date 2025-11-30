import os
import json
import pickle
import tempfile
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
from torchtext.vocab import Vocab

# 假设 GeneVocab 类定义在 gene_tokenizer.py 中
from gene_tokenizer import GeneVocab, tokenize_batch, pad_batch, tokenize_and_pad_batch, random_mask_value, _build_default_gene_vocab

def test_gene_vocab_initialization():
    """测试 GeneVocab 的初始化"""
    print("\n=== 测试 GeneVocab 初始化 ===")
    
    # 测试数据
    genes = ["TP53", "BRCA1", "EGFR", "MYC", "APOE", "BRCA1", "TP53", "TP53"]
    specials = ["<pad>", "<cls>", "<mask>"]
    
    # 1. 从基因列表初始化
    vocab = GeneVocab(genes, specials=specials)
    print("从基因列表初始化:")
    print(f"词汇表大小: {len(vocab)}")
    print(f"TP53 的索引: {vocab['TP53']} (应为最高频基因)")
    print(f"<pad> 的索引: {vocab['<pad>']} (应为0)")
    assert len(vocab) == len(set(genes)) + len(specials)
    assert vocab["TP53"] == 3  # 特殊token在前，TP53是第一个普通token
    
    # 2. 从现有 Vocab 对象初始化
    torch_vocab = Vocab(OrderedDict([("TEST", 1)]))
    vocab2 = GeneVocab(torch_vocab)
    print("\n从现有 Vocab 对象初始化:")
    print(f"词汇表大小: {len(vocab2)}")
    print(f"TEST 的索引: {vocab2['TEST']}")
    assert vocab2["TEST"] == 0
    
    # 3. 测试特殊token位置
    vocab3 = GeneVocab(genes, specials=specials, special_first=False)
    print("\n特殊token在末尾:")
    print(f"<pad> 的索引: {vocab3['<pad>']} (应为 {len(set(genes))})")
    assert vocab3["<pad>"] == len(set(genes))

def test_gene_vocab_file_io():
    """测试 GeneVocab 的文件读写功能"""
    print("\n=== 测试 GeneVocab 文件读写 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 测试数据
        genes = ["TP53", "BRCA1", "EGFR"]
        specials = ["<pad>", "<cls>"]
        vocab = GeneVocab(genes, specials=specials)
        
        # 1. 测试 JSON 格式
        json_path = tmp_path / "test_vocab.json"
        vocab.save_json(json_path)
        print(f"保存到 JSON: {json_path}")
        
        loaded_json = GeneVocab.from_file(json_path)
        print("从 JSON 加载:")
        print(f"词汇表大小: {len(loaded_json)}")
        assert loaded_json.get_stoi() == vocab.get_stoi()
        
        # 2. 测试 PKL 格式
        pkl_path = tmp_path / "test_vocab.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"\n保存到 PKL: {pkl_path}")
        
        loaded_pkl = GeneVocab.from_file(pkl_path)
        print("从 PKL 加载:")
        print(f"词汇表大小: {len(loaded_pkl)}")
        assert loaded_pkl.get_stoi() == vocab.get_stoi()
        
        # 3. 测试从字典创建
        test_dict = {"GENE1": 0, "GENE2": 1, "<pad>": 2}
        vocab_from_dict = GeneVocab.from_dict(test_dict)
        print("\n从字典创建:")
        print(f"词汇表: {vocab_from_dict.get_stoi()}")
        assert vocab_from_dict["GENE2"] == 1

def test_gene_vocab_special_tokens():
    """测试 GeneVocab 的特殊token处理"""
    print("\n=== 测试特殊token处理 ===")
    
    genes = ["TP53", "BRCA1"]
    specials = ["<pad>", "<cls>", "<mask>"]
    
    # 1. 测试 pad_token 设置
    vocab = GeneVocab(genes, specials=specials)
    vocab.pad_token = "<pad>"
    print(f"设置 pad_token: {vocab.pad_token}")
    assert vocab.pad_token == "<pad>"
    
    # 2. 测试 set_default_token
    vocab.set_default_token("<pad>")
    print(f"默认索引: {vocab.get_default_index()} (应为 {vocab['<pad>']})")
    assert vocab.get_default_index() == vocab["<pad>"]
    
    # 3. 测试特殊token在构建时的处理
    print("\n测试构建时特殊token处理:")
    vocab2 = GeneVocab(genes + ["<pad>"], specials=specials)
    print(f"<pad> 的索引: {vocab2['<pad>']} (应为0)")
    assert "<pad>" in vocab2
    assert vocab2["<pad>"] == 0

def test_tokenization_functions():
    """测试 tokenize 和 pad 相关函数"""
    print("\n=== 测试 tokenize 和 pad 函数 ===")
    
    # 创建词汇表
    genes = ["TP53", "BRCA1", "EGFR", "MYC"]
    specials = ["<pad>", "<cls>", "<mask>"]
    vocab = GeneVocab(genes, specials=specials)
    
    # 准备测试数据
    gene_ids = np.array([vocab[g] for g in genes])
    data = np.array([
        [1.0, 0.0, 2.0, 0.5],  # TP53:1.0, BRCA1:0.0, EGFR:2.0, MYC:0.5
        [0.0, 3.0, 0.0, 1.5]   # TP53:0.0, BRCA1:3.0, EGFR:0.0, MYC:1.5
    ])
    
    # 1. 测试 tokenize_batch
    tokenized = tokenize_batch(
        data, 
        gene_ids,
        append_cls=True,
        include_zero_gene=False
    )
    print("\ntokenize_batch 结果:")
    for i, (genes, values) in enumerate(tokenized):
        gene_names = [vocab.lookup_token(g) for g in genes.tolist()]
        print(f"样本 {i}: 基因: {gene_names}, 值: {values.tolist()}")
    
    # 2. 测试 pad_batch
    padded = pad_batch(
        tokenized,
        max_len=5,
        vocab=vocab,
        pad_token="<pad>"
    )
    print("\npad_batch 结果:")
    print(f"基因ID形状: {padded['genes'].shape}")
    print(f"值形状: {padded['values'].shape}")
    
    # 3. 测试 tokenize_and_pad_batch
    combined = tokenize_and_pad_batch(
        data,
        gene_ids,
        max_len=5,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=0,
        append_cls=True
    )
    print("\ntokenize_and_pad_batch 结果:")
    print(f"基因ID: {combined['genes']}")
    print(f"值: {combined['values']}")
    
    # 4. 测试 random_mask_value
    masked = random_mask_value(
        combined["values"],
        mask_ratio=0.3,
        mask_value=-1
    )
    print("\nrandom_mask_value 结果:")
    print(f"掩码后的值: {masked}")

def test_default_gene_vocab():
    """测试默认基因词汇表构建"""
    print("\n=== 测试默认基因词汇表 ===")
    
    # 创建临时目录和模拟数据
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_file = tmp_path / "human.gene_name_symbol.from_genenames.org.tsv"
        
        # 创建模拟数据文件
        mock_data = "Approved symbol\nTP53\nBRCA1\nEGFR\nMYC\nAPOE"
        with open(mock_file, "w") as f:
            f.write(mock_data)
        
        # 测试构建默认词汇表
        vocab = _build_default_gene_vocab(
            download_source_to=str(tmp_path),
            save_vocab_to=tmp_path / "default_vocab.json"
        )
        
        print(f"构建的默认词汇表大小: {len(vocab)}")
        print(f"包含基因示例: TP53={vocab['TP53']}, BRCA1={vocab['BRCA1']}")
        
        assert "TP53" in vocab
        assert "BRCA1" in vocab
        assert "<pad>" in vocab  # 即使未指定，也应包含特殊token

if __name__ == "__main__":
    # 运行所有测试
    test_gene_vocab_initialization()
    test_gene_vocab_file_io()
    test_gene_vocab_special_tokens()
    test_tokenization_functions()
    test_default_gene_vocab()
    print("\n所有测试完成!")