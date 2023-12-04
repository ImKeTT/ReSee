#!/usr/bin/env python
#-*- coding: utf-8 -*-
 #Enter features here
import json, torch
import random

import numpy as np
from collections import OrderedDict
import os, sys, math, time, copy

NUM_CONTEXT_SPECIAL_TOKEN=0

def jsonload(file):
    outs = []
    for line in open(file, 'r'):
        outs.append(json.loads(line))
    return outs

def load_pt_list(pt_list):
    img_feats = ()
    for ii in pt_list:
        img_feats = img_feats + (torch.load(ii),)
    img_feats = torch.cat(img_feats, dim=0)
    return img_feats


def sort_config(config, args):
    config.use_img_layernorm = args.use_img_layernorm
    config.img_feature_type = args.img_feature_type
    config.img_layer_norm_eps = args.img_layer_norm_eps
    config.img_dim = args.img_feature_dim
    config.hidden_dropout_prob = args.drop_out
    config.label_smoothing = args.label_smoothing
    config.img_add_pos = args.img_add_pos
    config.no_turn_vis = args.no_turn_vis  ## add turn-level visual knowledge
    config.no_ent_vis = args.no_ent_vis ## add entity-level visual knowledge
    config.context_length = args.max_seq_length + NUM_CONTEXT_SPECIAL_TOKEN
    # config.code_voc = args.code_voc
    # config.code_dim = args.code_dim

    return config

def _load_weights_pretrained_gpt2(m_lm, pm_lm, share_para=True):
    """

    :param m: MMDialog GPT2 Model
    :param pm: GPT2LMHeadModel
    :param share_para:
    :return:
    """
    pm = pm_lm.transformer
    m = m_lm.transformer
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)

    m_lm.lm_head.weight = pm_lm.lm_head.weight

def rearrange_unilm_weights(model_state):
    output_state = OrderedDict()
    for key in model_state.keys():
        key_list = key.split(".")
        if key_list[0] == "cls":
            if key == "cls.predictions.bias":
                tmp_key = "bias"
            elif key_list[1] != "seq_relationship":
                tmp_key = ".".join(key_list[2:])
            else:
                tmp_key = key
        else:
            tmp_key = key
        output_state[tmp_key] = model_state[key]

    return output_state


def tsv_writer(values, tsv_file_name, sep='\t'):
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    with open(tsv_file_name_tmp, 'wb') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
            v = v.encode()
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)

def gen_random_word_list(tokenizer, vocab_size):
    """
    remove special tokens for random word replacment
    :param tokenizer:
    :param vocab_size:
    :return:
    """
    wordlist = list(range(138, vocab_size))
    return wordlist

def gen_random_word_list_t5(tokenizer, vocab_size):
    """
    remove special tokens for random word replacment
    :param tokenizer:
    :param vocab_size:
    :return:
    """
    wordlist = list(range(vocab_size))
    wordlist.remove(tokenizer.bos_token_id)
    wordlist.remove(tokenizer.eos_token_id)
    wordlist.remove(tokenizer.mask_token_id)
    wordlist.remove(tokenizer.pad_token_id)
    wordlist.remove(tokenizer.unk_token_id)
    wordlist.remove(tokenizer.sep_token_id)
    for token in tokenizer.additional_special_tokens:
        wordlist.remove(tokenizer.convert_tokens_to_ids(token))
    return wordlist

def setup_seed(seed=3407):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False