#!/usr/bin/env python3
# coding: utf-8

import torch
import re
from ricefm.model.model_main import TransformerModel
from ricefm.tokenizer.gene_tokenizer import GeneVocab


def load_model(vocab):
    """
    Loads the Transformer model with specified parameters.

    Returns:
        TransformerModel: The initialized Transformer model.
    """
    ntokens = len(vocab)
    model_param = {
        'ntoken': ntokens,
        'd_model': 64,
        'nhead': 4,
        'd_hid': 64,
        'nlayers': 4,
        'nlayers_cls': 1,
        'n_cls': 1,
        'dropout': 0.5,
        'pad_token': "<pad>",
        'do_mvc': False,
        'do_dab': False,
        'use_batch_labels': False,
        'num_batch_labels': None,
        'domain_spec_batchnorm': False,
        'input_emb_style': "continuous",
        'cell_emb_style': "cls",
        'mvc_decoder_style': "inner product",
        'ecs_threshold': 0.3,
        'explicit_zero_prob': False,
        'fast_transformer_backend': "flash",
        'pre_norm': False,
        'vocab': vocab,
        'pad_value': -2,
        'n_input_bins': 51,
        'use_fast_transformer': True,
    }
    model = TransformerModel(**model_param)
    return model


def load_pretrain_model(model_file, model, load_param_prefixs=None):
    """
    Loads pre-trained weights into the model, selectively loading parameters that match
    the current model's architecture.

    Args:
        model_file (str): Path to the file containing pre-trained model weights.
        model (torch.nn.Module, optional): Model instance to load weights into.
            If None, uses the class's model attribute.
        load_param_prefixs (list of str, optional): List of parameter prefixes to selectively load.

    Returns:
        torch.nn.Module: The model with updated weights from the pre-trained file.

    Raises:
        FileNotFoundError: If the model file is not found.
    """

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location='cpu')
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    pretrained_dict = {re.sub(r'module.', '', k): v for k, v in pretrained_dict.items()}
    if load_param_prefixs is not None:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in load_param_prefixs])
        }
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def load_vocab(path):
    """
    Loads the gene vocabulary from a file and adds special tokens.

    Returns:
        GeneVocab: The loaded gene vocabulary.
    """
    vocab = GeneVocab.from_file(path)
    special_tokens = ['<pad>', '<cls>', '<eoc>']
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab['<pad>'])
    return vocab


if __name__ == '__main__':
    vocab = load_vocab("/home/share/huadjyin/home/s_qiuping1/workspace/plant/scGPT/run/demo/vocab.json")
    model = load_model(vocab)
    path = '/home/share/huadjyin/home/s_qiuping1/workspace/plant/scGPT/run/save/eval-Mar05-14-05-2025/best_model.pt'
    model = load_pretrain_model(path, model)
