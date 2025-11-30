#!/usr/bin/env python3
# coding: utf-8

import anndata as ad
import numpy as np
import json

with open('/home/share/huadjyin/home/s_qiuping1/workspace/plant/data/vocab.json', 'r') as fd:
    gene_vocab = json.load(fd)
    genes = list((gene_vocab.keys()))


def filter_large_anndata(input_file, output_file, gene_list):
    """
    读取一个大 h5ad 文件，仅保留基因列表中的基因，避免内存溢出
    """
    adata = ad.read_h5ad(input_file, backed="r")  # 只读模式
    gene_mask = np.isin(adata.var_names, gene_list)  # 生成基因筛选掩码

    # 创建新的 AnnData 仅存储筛选后的数据
    adata_filtered = ad.AnnData(
        X=adata[:, gene_mask].X,  # 仅保留筛选的基因
        obs=adata.obs.copy(),
        var=adata.var.iloc[gene_mask].copy()
    )

    adata_filtered.write_h5ad(output_file)  # 保存过滤后的数据


path = '/home/share/huadjyin/home/s_qiuping1/workspace/plant/zh11_rice_data/OMIX489/OMIX489-20-01.h5ad'
output = '/home/share/huadjyin/home/s_qiuping1/workspace/plant/data/pretrain_data/OMIX489.h5ad'
filter_large_anndata(path, output, genes)
