#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@file: dialog_reader.py
@author: ImKe at 2022/6/7
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

import os
import copy
import random
import numpy as np
from tqdm import tqdm
from data_utils.eda import *
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

# from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask
# from src.data_utils.data_reader import getDataLoader

class MMDialogDataset(Dataset):
    def __init__(self, dialog_episodes, device, entity_turn, caption_turn, entities_img_feats_ids,
                 turn_img_feats_ids, turn_img_feats, ent_img_feats, img_add_pos, img_dim,
                 history_in_context=True, max_episode_length=1, tokenizer=None, max_turn_img_seq_length=3,
                 per_max_turn_img_seq_length=20, max_ent_img_seq_length=8, max_seq_length=256,
                 max_seq_a_length=40, is_train=True,  mask_prob=0.15, mask_response_prob=0.5, response_sep="[SEP_0]",
                 context_sep="[SEP_1]", caption_sep="[SEP_2]", random_token_list=None,
                 max_masked_response_tokens=30,  max_masked_context_tokens=50, test=False, dataset="WOW",
                 add_cap=True, add_ent=True, seq2seq=False, no_ent_vis=False, no_turn_vis=False, EDA=False,
                 eda_pos='res', num_aug=5, knowledge_len=0, use_gpt=False, ent_img_num=1):
        """
        Data Constructor
        :param dialog_episodes: raw wow dataset
        :param device:
        :param entity_turn:
        :param caption_turn:
        :param entities_img_feats_ids:
        :param turn_img_feats_ids:
        :param history_in_context:
        :param max_episode_length:
        :param tokenizer:
        :param max_turn_img_seq_length:
        :param max_ent_img_seq_length:
        :param max_seq_length:
        :param max_seq_a_length:
        :param is_train:
        :param mask_prob:
        :param mask_response_prob:
        :param context_sep:
        :param caption_sep:
        :param num_special_tokens:
        :param max_masked_response_tokens:
        :param max_masked_context_tokens:
        """
        self.dataset = dataset
        self.add_cap = add_cap
        self.add_ent = add_ent
        self.no_ent_vis = no_ent_vis
        self.no_turn_vis = no_turn_vis
        self.eda = EDA
        self.use_gpt=use_gpt
        self.knowledge_len=knowledge_len
        self.episodes = self._preprocess_wow_episodes(dialog_episodes, entity_turn, caption_turn, entities_img_feats_ids,
                                                      turn_img_feats_ids, history_in_context, test, max_episode_length)
        self.seq2seq = seq2seq
        if self.seq2seq:
            self.tensorizer = MMTensorizerSeq2Seq(tokenizer, device, img_dim=img_dim, max_turn_img_seq_length=max_turn_img_seq_length,
                                           per_max_turn_img_seq_length=per_max_turn_img_seq_length, max_ent_img_seq_length=max_ent_img_seq_length,
                                           max_seq_a_length=max_seq_a_length, max_seq_len=max_seq_length, mask_prob=mask_prob,
                                           mask_response_prob=mask_response_prob, max_masked_response_tokens=max_masked_response_tokens,
                                           max_masked_context_tokens=max_masked_context_tokens, random_token_list=random_token_list,
                                           is_train=is_train, EDA=EDA, num_aug=num_aug, knowledge_len=knowledge_len, eda_pos=eda_pos)
        elif self.use_gpt:
            self.tensorizer = MMTensorizerGPT(tokenizer, device, img_dim=img_dim, max_turn_img_seq_length=max_turn_img_seq_length,
                                           per_max_turn_img_seq_length=per_max_turn_img_seq_length, max_ent_img_seq_length=max_ent_img_seq_length,
                                           max_seq_a_length=max_seq_a_length, max_seq_len=max_seq_length, mask_prob=mask_prob,
                                           mask_response_prob=mask_response_prob, max_masked_response_tokens=max_masked_response_tokens,
                                           max_masked_context_tokens=max_masked_context_tokens, random_token_list=random_token_list,
                                           is_train=is_train, EDA=EDA, num_aug=num_aug, knowledge_len=knowledge_len)
        else:
            self.tensorizer = MMTensorizer(tokenizer, device, img_dim=img_dim, max_turn_img_seq_length=max_turn_img_seq_length,
                                           per_max_turn_img_seq_length=per_max_turn_img_seq_length, max_ent_img_seq_length=max_ent_img_seq_length,
                                           max_seq_a_length=max_seq_a_length, max_seq_len=max_seq_length, mask_prob=mask_prob,
                                           mask_response_prob=mask_response_prob, max_masked_response_tokens=max_masked_response_tokens,
                                           max_masked_context_tokens=max_masked_context_tokens, response_sep=response_sep,
                                           context_sep=context_sep, caption_sep=caption_sep, random_token_list=random_token_list,
                                           is_train=is_train, knowledge_len=knowledge_len)
        self.max_turn_img_seq_length = max_turn_img_seq_length
        self.max_ent_img_seq_length = max_ent_img_seq_length
        self.turn_img_feats = turn_img_feats
        self.ent_img_feats = ent_img_feats
        self.is_train = is_train
        self.cross_attn = True if img_add_pos!="concat" else False
        self.ent_img_num=ent_img_num


    def _preprocess_wow_episodes(self, episodes, entity_turn, caption_turn, entities_img_feats_ids,
                             turn_img_feats_ids, history_in_context, test=False, max_episode_length=1):
        """
        Tokenize all the fields in Wizard-of-Wikipedia.
        Return List[Dict[samples](episodes)]
        split to several episodes for one dialog session

        Output Example:
        [
            { # one episode
                'context': [], # in episode length
                'response': [],
                'title': [],
                'sample_id': int,
                'episode_num': int,
            }
            ...
            {
                # another episode
            }
        ]
        """

        if self.dataset == "WOW":
            dialog_key = "dialog"
        elif self.dataset == "DD":
            dialog_key = "dialogue"
            assert self.knowledge_len==0, "No knowledge document for Daily Dialogue dataset.."
        else:
            raise NotImplementedError

        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, desc="Preprocess episodes", ncols=100)):
            ## fetch the turn
            # start_id = int(len(episode[dialog_key]) / 2) - 1 if (test or self.use_gpt) else 1
            start_id = int(len(episode[dialog_key]) / 2) - 1 if test else 1
            ## avoid no response for training
            if int(len(episode[dialog_key]) / 2) - 1 <= 0:
                continue

            full_knowledge = ''
            if self.knowledge_len:
                for line in episode['chosen_topic_passage']:
                    full_knowledge += line.strip() + ' '

            singular_length = len(episode[dialog_key]) % 2

            for example_num in range(start_id, int(len(episode[dialog_key]) / 2)):
                new_examples = {'context': [],
                                'response': '',
                                'entities': [],
                                'captions': '',
                                'entity_img_feat_ids': [],
                                'turn_img_feat_ids': [],
                                'sample_id': example_num,
                                'episode_num': episode_num,
                                'knowledge': ''}


                history = []
                turn_entities = set()
                caption_history = []

                turn_level_img_fea_ids = []
                entity_level_img_fea_ids = []


                if example_num != 0 and history_in_context:
                    start_idx = max(0, example_num - max_episode_length)
                    for num in range(start_idx, example_num):
                        history.append(episode[dialog_key][num * 2]['text'].lower().strip()) ## context
                        history.append(episode[dialog_key][num * 2 + 1]['text'].lower().strip()) ## response

                        ## entity_turn is actually entity for every line in a dialogue
                        curr_entity_turn = entity_turn[episode_num][num * 2]
                        curr_entity_turn.extend(entity_turn[episode_num][num * 2 + 1])
                        for entity in curr_entity_turn:
                            turn_entities.add(entity)

                        caption_history.append(caption_turn[episode_num][num].lower().strip())
                        turn_level_img_fea_ids.append(turn_img_feats_ids[episode_num][num])

                # if singular_length and test:
                #     context = history + [episode[dialog_key][example_num * 2]['text'].lower().strip()] + \
                #               [episode[dialog_key][example_num * 2 + 1]['text'].lower().strip()]
                #     response = episode[dialog_key][(example_num+1) * 2]['text'].lower().strip()
                # else:
                context = history + [episode[dialog_key][example_num * 2]['text'].lower().strip()]
                response = episode[dialog_key][example_num * 2 + 1]['text'].lower().strip()

                ## Not adding the current turn-level image to avoid image leakage
                # if test:
                #     turn_level_img_fea_ids.append(turn_img_feats_ids[episode_num][example_num])

                ## exclude entities in the response entity_turn[episode_num][2 * example_num + 1]
                for entity in entity_turn[episode_num][2 * example_num]:
                    turn_entities.add(entity)
                turn_entities = list(turn_entities)
                for ent in turn_entities:
                    entity_level_img_fea_ids.append(entities_img_feats_ids[ent])

                ## not append the current caption to avoid info leakage
                captions = caption_history #+ [caption_turn[episode_num][example_num].lower().strip()]
                if test:
                    captions += [caption_turn[episode_num][example_num].lower().strip()]


                new_examples['context'] = context
                new_examples['response'] = response
                new_examples['entities'] = turn_entities
                new_examples['captions'] = captions
                new_examples['entity_img_feat_ids'] = entity_level_img_fea_ids
                new_examples['turn_img_feat_ids'] = turn_level_img_fea_ids
                new_examples['knowledge'] = full_knowledge

                new_episodes.append(new_examples)

        return new_episodes

    def get_image_features(self, img_idx, feats_file):
        return feats_file[img_idx]

    def __getitem__(self, idx):
        example = self.episodes[idx]
        text_a = example['response']
        if self.use_gpt:
            ## add eos for every dialogue turn
            text_b = "<|endoftext|>".join(example['context'])
            if self.is_train:
                text_b += "<|endoftext|>" + example['response']
        else:
            text_b = " ".join(example['context'])
        turn_img_feat = ()
        for img_id in example['turn_img_feat_ids']:
            turn_img_feat += (self.get_image_features(img_id, self.turn_img_feats).unsqueeze(0),)
        turn_img_feat = torch.cat(turn_img_feat, dim=0)

        if len(example['entity_img_feat_ids']):
            ent_img_feat = ()
            for img_id in example['entity_img_feat_ids']:
                for ind in range(self.ent_img_num):
                    ent_img_feat += (self.get_image_features(img_id, self.ent_img_feats[ind]).unsqueeze(0),)
            ent_img_feat = torch.cat(ent_img_feat, dim=0)
        else:
            ent_img_feat = torch.tensor([])

        text_c = example['captions'] if self.add_cap else None
        text_e = example['entities'] if self.add_ent else None
        text_k = example['knowledge'] if self.knowledge_len else None

        example_out = self.tensorizer.tensorize_example(text_a, turn_img_feat, ent_img_feat, text_b,
                                                        text_c, text_e, self.cross_attn, text_k=text_k,
                                                        no_turn_vis=self.no_turn_vis, no_ent_vis=self.no_ent_vis)
        example = {}
        if self.seq2seq:
            if self.is_train:
                example["decoder_input_ids"] = example_out[0]
                example["masked_decoder_input_ids"] = example_out[1]
                example["input_ids"] = example_out[2]
                example["decoder_attention_mask"] = example_out[3]
                example["attention_mask"] = example_out[4]
                example["segment_ids"] = example_out[5]
                example["turn_img_feats"] = example_out[6]
                example["turn_img_len"] = 0 if text_c is None else min(self.max_turn_img_seq_length, len(text_c))
                example["ent_img_feats"] = example_out[7]
                example["ent_img_len"] = 0 if text_e is None else min(self.max_ent_img_seq_length, len(text_e))
                example["masked_res_pos"] = example_out[8]
                example["masked_ids"] = example_out[9]
                example["cap_pos"] = example_out[10]
                example["ent_pos"] = example_out[11]
                example["ent_img_pos"] = example_out[12]
                if not self.no_ent_vis and text_e is not None:
                    example["ent_ids"] = example_out[13]
                if self.eda:
                    example["input_ids_eda"] = example_out[-1]
                # for i, ii in enumerate(example_out):
                #     if i != 4:
                #         if ii is None:
                #             print(i, i, i)
                #         print(i, ii.shape)
                    # else:
                    #     print(ii[0].shape, ii[1].shape)
            else:
                example["decoder_input_ids"] = example_out[0]
                example["input_ids"] = example_out[1]
                example["decoder_attention_mask"] = example_out[2]
                example["attention_mask"] = example_out[3]
                example["segment_ids"] = example_out[4]
                example["turn_img_feats"] = example_out[5]
                example["turn_img_len"] = 0 if text_c is None else min(self.max_turn_img_seq_length, len(text_c))
                example["ent_img_feats"] = example_out[6]
                example["ent_img_len"] = 0 if text_e is None else min(self.max_ent_img_seq_length, len(text_e))
                example["cap_pos"] = example_out[7]
                example["ent_pos"] = example_out[8]
                example["ent_img_pos"] = example_out[9]
                example["response"] = text_a

        # input_ids, attention_mask, segment_ids, turn_img_feat, ent_img_feat, cap_pos, ent_pos, ent_img_pos, input_ids_eda
        elif self.use_gpt:
            example["input_ids"] = example_out[0]
            example["attention_mask"] = example_out[1]
            example["segment_ids"] = example_out[2]
            example["turn_img_feats"] = example_out[3]
            example["turn_img_len"] = 0 if text_c is None else min(self.max_turn_img_seq_length, len(text_c))
            example["ent_img_feats"] = example_out[4]
            example["ent_img_len"] = 0 if text_e is None else min(self.max_ent_img_seq_length, len(text_e))
            example["ent_pos"] = example_out[6]
            if self.eda and self.is_train:
                example["input_ids_eda"] = example_out[-2]
            if not self.is_train:
                example["response"] = text_a
            ## use for cross attention in multimodal GPT
            example["ent_ids"] = example_out[-1]
        ## UniLM input
        else:
            ## is_train
            if self.is_train:
                example["input_ids"] = example_out[0]
                example["attention_mask"] = example_out[1]
                example["segment_ids"] = example_out[2]
                example["turn_img_feats"] = example_out[3]
                example["turn_img_len"] = 0 if text_c is None else min(self.max_turn_img_seq_length, len(text_c))
                example["ent_img_feats"] = example_out[4]
                example["ent_img_len"] = 0 if text_e is None else min(self.max_ent_img_seq_length, len(text_e))
                example["masked_pos"] = example_out[5]
                example["masked_ids"] = example_out[6]
                example["cap_pos"] = example_out[7]
                example["ent_pos"] = example_out[8]
                example["input_ids_rmask"] = example_out[9]
                example["masked_res"] = example_out[10]
                example["masked_res_pos"] = example_out[11]
                example["ent_img_pos"] = example_out[12]
                example["ent_ids"] = example_out[13]

                # for i, ii in enumerate(example_out):
                #     if i != 1:
                #         if ii is None:
                #             print(i, i, i)
                #         print(i, ii.shape)
                    # else:
                    #     print(ii[0].shape, ii[1].shape)
            else:
                example["input_ids"] = example_out[0]
                example["attention_mask"] = example_out[1]
                example["segment_ids"] = example_out[2]
                example["turn_img_feats"] = example_out[3]
                example["turn_img_len"] = 0 if text_c is None else min(self.max_turn_img_seq_length, len(text_c))
                example["ent_img_feats"] = example_out[4]
                example["ent_img_len"] = 0 if text_e is None else min(self.max_ent_img_seq_length, len(text_e))
                example["masked_pos"] = example_out[5]
                example["gth_ids"] = example_out[6]
                example["ent_labels_start_posid"] = example_out[7]
                example["cap_start_posid"] = example_out[8]
                example["context_start_posid"] = example_out[9]
                example["context_end_posid"] = example_out[10]
                example["response"] = text_a


        return example

    def __len__(self):
        return len(self.episodes)


class MMTensorizer(object):
    def __init__(self, tokenizer, device, img_dim, max_turn_img_seq_length=5, per_max_turn_img_seq_length=20,
                 max_ent_img_seq_length=8, max_seq_a_length=30, max_seq_len=100, mask_prob=0.15, mask_response_prob=0.5,
                 max_masked_response_tokens=10, max_masked_context_tokens=20, response_sep="[SEP_1]", context_sep="[SEP_1]",
                 caption_sep="[SEP_2]", random_token_list=None, is_train=True, knowledge_len=0):
        """Constructor.
        :param tokenizer: text tokenizer
        :param device:
        :param max_turn_img_seq_length: max turn level image number
        :param max_ent_img_seq_length: max entity number / entity level image number
        :param max_seq_a_length: max response sequence length
        :param max_seq_len: max total textual sequence length
        :param mask_prob: mask prob for
        :param mask_response_prob: response mask prob
        :param max_masked_tokens:
        :param context_sep: tokenizer special token
        :param caption_sep: tokenizer special token
        :param num_special_tokens:
        :param is_train:
        """
        self.tokenizer = tokenizer
        self.device = device
        self.is_train = is_train
        self.max_turn_img_seq_len = max_turn_img_seq_length
        self.per_max_turn_img_seq_len = per_max_turn_img_seq_length
        self.max_ent_img_seq_len = max_ent_img_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_response_prob = mask_response_prob
        self.max_masked_response_tokens = max_masked_response_tokens
        self.max_masked_context_tokens = max_masked_context_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))

        self.context_sep_token = context_sep
        self.caption_sep_token = caption_sep
        self.response_sep_token = response_sep
        self.random_token_list = random_token_list
        self.img_dim = img_dim

    def random_word(self, tokens, mask_prob, candidate_masked_idx, seq_len, max_masked_tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        """
        tokens_ = copy.copy(tokens)
        num_masked = min(max(round(mask_prob * seq_len), 1), max_masked_tokens)
        num_masked = int(num_masked)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        masked_token = [tokens_[i] for i in masked_idx]
        for pos in masked_idx:
            if random.random() <= 0.8:
                # 80% chance to be a ['MASK'] token
                tokens_[pos] = self.tokenizer.mask_token
            elif random.random() <= 0.5:
                # 10% chance to be a random word ((1-0.8)*0.5)
                from random import choice
                i = choice(self.random_token_list)
                self.tokenizer._convert_id_to_token(i)
                tokens_[pos] = self.tokenizer._convert_id_to_token(i)
            else:
                # 10% chance to remain the same (1-0.8-0.1)
                pass
        return tokens_, masked_token, masked_idx, num_masked

    def mask_word(self, tokens, candidate_masked_idx):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        """
        token_ = copy.copy(tokens)
        masked_token = [token_[i] for i in candidate_masked_idx]
        for pos in candidate_masked_idx:
            token_[pos] = self.tokenizer.mask_token
        return token_, masked_token

    def pad_img_feat(self, img_feat, max_img_seq_len, img_dim):
        img_len = img_feat.size(0)
        if img_len > max_img_seq_len:
            img_feat = img_feat[0: max_img_seq_len, ]
            img_len = img_feat.size(0)
        else:
            padding_matrix = torch.zeros((max_img_seq_len - img_len, img_dim), device=self.device)
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        return img_feat, img_len

    def tensorize_example(self, text_a, turn_img_feat, ent_img_feat, text_b=None, text_c=None, text_e=None, cross_attn=True,
                          bos_token_segment_id=0, cls_token_segment_id=0, pad_token_segment_id=0, eos_token_segment_id=0,
                          sequence_a_segment_id=1, sequence_b_segment_id=2, sequence_c_segment_id=3, sequence_e_segment_id=4,
                          no_turn_vis=False, no_ent_vis=False, text_k=None):
        """
        convert data example to torch tensor

        :param text_a: response
        :param img_feat: 1xfeatsize + 1 + nxfeatsize: [turn-level image features] 'SEP' [entity level images]
        :param text_b: context
        :param text_c: captions
        :param text_e: entities
        :param cls_token_segment_id:
        :param pad_token_segment_id:
        :param sequence_a_segment_id:
        :param sequence_b_segment_id:
        :param sequence_e_segment_id:
        :param no_turn_vis: no turn-level visual knowledge
        :param no_ent_vis: no entity-level visual knowledge
        :return:
        For training:

        input_ids is like:
        [BOS] Dialog Response [SEP] ..[PAD] [CLS] Dialog Context (with [MASK]) [CONSEP] Caption (with [MASK]) [CAPSEP] Entities [EOS] [PAD]...[PAD]

        input_ids_rmask is like:
        [BOS] [MASK]...[MASK] [SEP] .. [PAD] [CLS] Dialog Context [CONSEP] Caption [CAPSEP] Entities [EOS] [PAD]...[PAD]
        """
        no_vis = no_turn_vis and no_ent_vis
        if self.is_train:
            ## 1 for bos_token, 1 for sep_token
            ## tokenize without padding: make sure that every sentence
            ## begins with bos_token and ends with eos_token
            tokens_a = self.tokenizer.tokenize(text_a)
            if len(tokens_a) > self.max_seq_a_len - 2:
                tokens_a = tokens_a[:(self.max_seq_a_len - 2)]
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)

            gth_tokens_a = self.tokenizer.tokenize(text_a)
            if len(tokens_a) > self.max_seq_a_len - 2:
                tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

            if len(gth_tokens_a) > self.max_seq_a_len:
                gth_tokens_a = gth_tokens_a[:(self.max_seq_a_len)]

            padding_gth_len = self.max_seq_a_len - len(gth_tokens_a)
            gth_tokens_a += [self.tokenizer.pad_token] * padding_gth_len
            gth_ids = self.tokenizer.convert_tokens_to_ids(gth_tokens_a)
            gth_ids = torch.tensor(gth_ids, dtype=torch.long)

        tokens = [self.tokenizer.bos_token] + tokens_a + [self.response_sep_token]
        segment_ids = [bos_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)

        assert text_b is not None, "Dialog context should not be None.."
        # pad text_a to keep it in fixed length for better inference.
        ## padding response
        padding_a_len = self.max_seq_a_len - seq_a_len
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        if self.is_train:
            gth_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            gth_ids = torch.tensor(gth_ids, dtype=torch.long, device=self.device)

        ## process text_b
        EXTRA_TOKEN_LEN = 2 ## extra special tokens of context (text_b)
        cap_seq_len = 0 if text_c is None else (self.max_turn_img_seq_len * self.per_max_turn_img_seq_len)
        ent_seq_len = 0 if text_e is None else self.max_ent_img_seq_len
        max_b_seq_len = self.max_seq_len - cap_seq_len - ent_seq_len - \
                        self.max_seq_a_len - EXTRA_TOKEN_LEN
        b_start = len(tokens)
        max_b_end = b_start + max_b_seq_len + EXTRA_TOKEN_LEN
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_b_len = len(tokens_b)
        if tokens_b_len < max_b_seq_len:
            b_end = len(tokens) + tokens_b_len
            b_padding_len = max_b_seq_len - tokens_b_len
            tokens_b += ([self.tokenizer.pad_token] * b_padding_len)
            segment_ids += [cls_token_segment_id] + [sequence_b_segment_id] * tokens_b_len + [pad_token_segment_id] * b_padding_len
        else:
            tokens_b = tokens_b[: max_b_seq_len]
            segment_ids += [cls_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
            b_end = len(tokens) + len(tokens_b)
        tokens += [self.tokenizer.cls_token] + tokens_b + [self.context_sep_token]
        segment_ids += [sequence_b_segment_id]

        cap_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        if text_c is not None and not no_turn_vis:
            cap_idx = []
            ## process text_c captions
            c_start = len(tokens)
            c_end = c_start + self.max_turn_img_seq_len * self.per_max_turn_img_seq_len
            if len(text_c) < self.max_turn_img_seq_len:
                turn_img_seq_padding_len = self.max_turn_img_seq_len - len(text_c)
            else:
                text_c = text_c[: self.max_turn_img_seq_len]
                turn_img_seq_padding_len = 0
            c_start_tmp = c_start
            for turn_cap in text_c:
                tokens_c = self.tokenizer.tokenize(turn_cap)
                tokens_c_len = len(tokens_c)
                if tokens_c_len < self.per_max_turn_img_seq_len - 1: ## exclude caption sep special token
                    c_end_tmp = len(tokens) + tokens_c_len
                    c_padding_len = self.per_max_turn_img_seq_len - len(tokens_c) - 1
                    tokens_c += ([self.tokenizer.pad_token] * c_padding_len)
                    segment_ids += [sequence_c_segment_id] * tokens_c_len + [pad_token_segment_id] * c_padding_len
                else:
                    tokens_c = tokens_c[: self.per_max_turn_img_seq_len - 1]
                    segment_ids += [sequence_c_segment_id] * (self.per_max_turn_img_seq_len - 1)
                    c_end_tmp = len(tokens) + self.per_max_turn_img_seq_len - 1
                cap_idx.extend(list(range(c_start_tmp, c_end_tmp)))
                tokens += tokens_c + [self.caption_sep_token]
                segment_ids += [sequence_c_segment_id]
                c_start_tmp = len(tokens)
            tokens += ([self.tokenizer.pad_token] * (self.per_max_turn_img_seq_len - 1) + [self.caption_sep_token]) * turn_img_seq_padding_len
            segment_ids += ([pad_token_segment_id] * (self.per_max_turn_img_seq_len - 1) + [sequence_c_segment_id]) * turn_img_seq_padding_len
            cap_pos[cap_idx] = 1
        else:
            c_start = None
            c_end = None
            cap_idx = None

        if text_e is not None:
            ## process text_e entities
            e_start = len(tokens)
            text_e = " ".join(text_e)
            tokens_e = self.tokenizer.tokenize(text_e)
            tokens_e_len = len(tokens_e)
            ## prepare for multimodal entity regression
            if tokens_e_len < self.max_ent_img_seq_len - 1: ## exclude eos special token
                e_padding_len = self.max_ent_img_seq_len - tokens_e_len - 1
                tokens_e += ([self.tokenizer.pad_token] * e_padding_len)
                segment_ids += [sequence_e_segment_id] * tokens_e_len + [pad_token_segment_id] * e_padding_len
            else:
                tokens_e = tokens_e[: self.max_ent_img_seq_len - 1]
                segment_ids += [sequence_e_segment_id] * (self.max_ent_img_seq_len - 1)

            tokens += tokens_e + [self.tokenizer.eos_token]
            ## because there may be no entity for dialog turn, e_end includes padding tokens
            e_end = len(tokens) - 1
            segment_ids += [eos_token_segment_id]
            ent_ids = self.tokenizer.convert_tokens_to_ids(tokens_e)
        else:
            e_start = e_end = 0
            ent_ids = None
            # e_start = len(tokens)
            # tokens_e = [self.tokenizer.pad_token] * (self.max_ent_img_seq_len - 1) ## exclude eos token
            # tokens += tokens_e + [self.tokenizer.eos_token]
            # e_end = len(tokens) - 1
            # segment_ids += [pad_token_segment_id] * (self.max_ent_img_seq_len - 1) + [eos_token_segment_id]
            # ent_ids = self.tokenizer.convert_tokens_to_ids(tokens_e)

        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            masked_res_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            masked_ent_pos = torch.zeros(self.max_seq_len, dtype=torch.int)

            ## mask response
            candidate_masked_idx = list(range(1, seq_a_len))
            tokens_rmask, _ = self.mask_word(tokens, candidate_masked_idx)
            masked_res_pos[candidate_masked_idx] = 1

            # -----------------------------mask response part-----------------------------

            # randomly mask words for prediction, ignore [BOS]
            candidate_masked_idx = list(range(1, seq_a_len))
            random.shuffle(candidate_masked_idx)
            tokens, masked_token, masked_idx, num_masked = self.random_word(tokens, self.mask_response_prob,
                                                                            candidate_masked_idx, seq_a_len,
                                                                            self.max_masked_response_tokens)
            masked_pos[masked_idx] = 1

            # -----------------------------mask context part-----------------------------
            candidate_masked_idx_text_b = list(range(b_start+1, b_end)) ## cls token before text_b
            if cap_idx is not None:
                candidate_masked_idx_text_b.extend(cap_idx)
                # list(range(self.max_seq_a_len, tag_start))  # mask context
            ## mask context
            random.shuffle(candidate_masked_idx_text_b)
            tokens, masked_token_tb, masked_idx_tb, num_masked_tb = self.random_word(tokens, self.mask_prob, candidate_masked_idx_text_b,
                                                                                     len(candidate_masked_idx_text_b), self.max_masked_context_tokens)

            #  MRM with MLM in the same time
            masked_pos[masked_idx_tb] = 1
            # pad masked tokens to the same length
            total_num_masked = (num_masked + num_masked_tb)
            max_masked_tokens = self.max_masked_response_tokens + self.max_masked_context_tokens
            if total_num_masked < max_masked_tokens:
                masked_token = masked_token + masked_token_tb + ([self.tokenizer.pad_token] *
                                                                 (max_masked_tokens - total_num_masked))
            else:
                masked_token = masked_token + masked_token_tb

            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)

            # -----------------------------mask entity  part-----------------------------
            # else:

            ## NOT mask entity
            # candidate_masked_idx_ent = list(range(e_start, e_end))  # mask text_b
            # candidate_masked_idx_ent = [min(i, len(tokens) - 1) for i in candidate_masked_idx_ent]
            # random.shuffle(candidate_masked_idx_ent)
            # tokens, masked_token_ent, masked_idx_ent, num_masked_ent = self.random_word(tokens, self.mask_prob, candidate_masked_idx_ent,
            #                                                                          len(candidate_masked_idx_ent))

            # masked_ent_pos[masked_idx_ent] = 1.
            # # pad masked tokens to the same length
            # if num_masked_ent < self.max_masked_tokens:
            #     masked_token_ent = masked_token_ent + ([self.tokenizer.pad_token] *
            #                                            (self.max_masked_tokens - num_masked_ent))
            # masked_ent_ids = self.tokenizer.convert_tokens_to_ids(masked_token_ent)

            input_ids_rmask = self.tokenizer.convert_tokens_to_ids(tokens_rmask)

        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        turn_img_feat, ent_img_feat = turn_img_feat.to(self.device), ent_img_feat.to(self.device)
        turn_img_feat, turn_img_len = self.pad_img_feat(turn_img_feat, self.max_turn_img_seq_len, self.img_dim)
        ent_img_feat, ent_img_len = self.pad_img_feat(ent_img_feat, self.max_ent_img_seq_len, self.img_dim)

        if not no_vis:
            # prepare attention mask:
            # note that there is no attention from caption to image
            # because otherwise it will violate the triangle attention
            # for caption as caption will have full attention on image.
            seq_len = len(tokens)
            img_seq_len = 0
            if not no_turn_vis:
                img_seq_len += self.max_turn_img_seq_len
            if not no_ent_vis:
                img_seq_len += self.max_ent_img_seq_len
            ent_img_pos = torch.zeros(img_seq_len, dtype=torch.int, device=self.device)

            if not cross_attn:
                """
              C 1 1 1|1 1|0 0
              o 1 1 1|1 1|0 0
              n 1 1 1|1 1|0 0
              t -----|---|---
              I 1 1 1|1 1|0 0
              m 1 1 1|1 1|0 0
              g -----|---|---
              R 1 1 1|1 1|1 0
              e 1 1 1|1 1|1 1
              s Cont |Img|Res
                """
                max_len = seq_len + img_seq_len
                attention_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=self.device)
                if not no_ent_vis:
                    prefix_length = 0 if no_turn_vis else self.max_turn_img_seq_len
                    ent_img_idx = list(range(prefix_length, (e_end - e_start) + prefix_length))
                    ent_img_pos[ent_img_idx] = 1
                else:
                    ent_img_pos=0
                # R: response, L: context+caption+entities, F: image feature
                r_start, r_end = 0, seq_a_len
                ## for context (caption, entity)
                l_start, l_end = self.max_seq_a_len, self.max_seq_len
                ## for image features
                f_start, f_end = self.max_seq_len, max_len

                ## full vision for context (caption, entity) and image features
                # attention_mask[:, l_start: f_end].fill_(1)
                # triangle mask for response to response r-r
                attention_mask[r_start: r_end, r_start: r_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
                # # full attention for L-L, F-F
                attention_mask[l_start: l_end, l_start: l_end] = 1
                attention_mask[f_start: f_end, f_start: f_end] = 1
                # full attention for R-L, R-F
                attention_mask[r_start: r_end, l_start: l_end] = 1
                attention_mask[r_start: r_end, f_start: f_end] = 1
                # full attention for L-F, F-L:
                attention_mask[l_start: l_end, f_start: f_end] = 1
                attention_mask[f_start: f_end, l_start: l_end] = 1

                attention_mask = [attention_mask]
            else:
                padded_img_seq_len = seq_len
                attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.long, device=self.device)
                # R: response, L: context+caption+entities
                r_start, r_end = 0, seq_a_len
                l_start, l_end = self.max_seq_a_len, seq_len
                # triangle mask for caption to caption C-C
                attention_mask[r_start: r_end, r_start: r_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
                # full attention for L-L
                attention_mask[l_start: l_end, l_start: l_end] = 1
                # full attention for C-L
                attention_mask[r_start: r_end, l_start: l_end] = 1

                img_attention_mask = torch.zeros((padded_img_seq_len, padded_img_seq_len), dtype=torch.long, device=self.device)
                ## if use cross attention, pad image feature length to the same size as max sequence length
                ent_img_feat, _ = self.pad_img_feat(ent_img_feat, padded_img_seq_len - self.max_turn_img_seq_len, self.img_dim)
                ti_start, ti_end = 0, turn_img_len
                ei_start, ei_end = self.max_turn_img_seq_len, self.max_turn_img_seq_len + ent_img_len

                ## [turn-image, entity-img]
                ## full vision for both types of images
                img_attention_mask[ti_start: ti_end, ti_start: ti_end] = 1
                img_attention_mask[ei_start: ei_end, ei_start: ei_end] = 1
                img_attention_mask[ti_start: ti_end, ei_start: ei_end] = 1
                img_attention_mask[ei_start: ei_end, ti_start: ti_end] = 1

                attention_mask = [attention_mask, img_attention_mask]
        else:
            """
            1 1 1|0 0 
            1 1 1|0 0
            1 1 1|0 0
            -----|---
            1 1 1|1 0
            1 1 1|1 1
            Cont |Res
            """
            img_seq_len = self.max_turn_img_seq_len + self.max_ent_img_seq_len
            ent_img_pos = torch.zeros(img_seq_len, dtype=torch.int, device=self.device)
            seq_len = len(tokens)
            attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.long, device=self.device)
            r_start, r_end = 0, seq_a_len
            l_start, l_end = self.max_seq_a_len, self.max_seq_len

            # triangle mask for response to response
            attention_mask[r_start: r_end, r_start: r_end].copy_(self._triangle_mask[r_start: r_end, r_start: r_end])
            # full attention for context-context
            attention_mask[l_start: l_end, l_start: l_end] = 1
            # full attention for contex-response
            attention_mask[r_start: r_end, l_start: l_end] = 1
            ## zero mask for response-context

            attention_mask = [attention_mask]

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=self.device)

        # used for get entity position
        ent_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_e:
        ent_idx = list(range(e_start, e_end))
        ent_pos[ent_idx] = 1

        # cap_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_c:
        #     cap_idx = list(range(c_start, c_end))
        #     cap_pos[cap_idx] = 1

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long, device=self.device)
            input_ids_rmask = torch.tensor(input_ids_rmask, dtype=torch.long, device=self.device)
            ent_ids = torch.tensor(ent_ids, dtype=torch.long, device=self.device) if ent_ids is not None else 0
            # masked_ent_ids = torch.tensor(masked_ent_ids, dtype=torch.long, device=self.device)

            return (input_ids, attention_mask, segment_ids, turn_img_feat, ent_img_feat,
                     masked_pos, masked_ids, cap_pos, ent_pos, input_ids_rmask, gth_ids,
                     masked_res_pos, ent_img_pos, ent_ids)
        else:
            if c_start is None:
                c_start = -1
            return (input_ids, attention_mask, segment_ids, turn_img_feat, ent_img_feat,
                    masked_pos, gth_ids, e_start, c_start, b_start, max_b_end)



class MMTensorizerSeq2Seq(object):
    def __init__(self, tokenizer, device, img_dim, max_turn_img_seq_length=5, per_max_turn_img_seq_length=20,
                 max_ent_img_seq_length=8, max_seq_a_length=30, max_seq_len=100, mask_prob=0, mask_response_prob=0.5,
                 max_masked_response_tokens=10, max_masked_context_tokens=0, random_token_list=None, is_train=True,
                 EDA=True, eda_pos='res', num_aug=9, knowledge_len=0):
        """ Dataset Constructor.
        :param tokenizer: text tokenizer
        :param device:
        :param max_turn_img_seq_length: max turn level image number
        :param max_ent_img_seq_length: max entity number / entity level image number
        :param max_seq_a_length: max response sequence length
        :param max_seq_len: max total textual sequence length
        :param mask_prob: mask prob for
        :param mask_response_prob: response mask prob
        :param max_masked_tokens:
        :param context_sep: tokenizer special token
        :param caption_sep: tokenizer special token
        :param num_special_tokens:
        :param is_train:
        """
        self.tokenizer = tokenizer
        self.device = device
        self.is_train = is_train
        self.conx_eda = EDA and eda_pos=="conx"
        self.res_eda = EDA and eda_pos=="res"
        self.num_aug = num_aug
        self.knowledge_len = knowledge_len
        self.max_turn_img_seq_len = max_turn_img_seq_length
        self.per_max_turn_img_seq_len = per_max_turn_img_seq_length
        self.max_ent_img_seq_len = max_ent_img_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_response_prob = mask_response_prob
        self.max_masked_response_tokens = max_masked_response_tokens
        self.max_masked_context_tokens = max_masked_context_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))
        self.random_token_list = random_token_list
        self.img_dim = img_dim

    def random_word(self, tokens, mask_prob, candidate_masked_idx, seq_len, max_masked_tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        """
        tokens_ = copy.copy(tokens)
        num_masked = min(max(round(mask_prob * seq_len), 1), max_masked_tokens)
        num_masked = int(num_masked)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        masked_token = [tokens_[i] for i in masked_idx]
        for pos in masked_idx:
            if random.random() <= 0.8:
                # 80% chance to be a ['MASK'] token
                tokens_[pos] = self.tokenizer.mask_token
            elif random.random() <= 0.5:
                # 10% chance to be a random word ((1-0.8)*0.5)
                from random import choice
                i = choice(self.random_token_list)
                tokens_[pos] = self.tokenizer._convert_id_to_token(i)
            else:
                # 10% chance to remain the same (1-0.8-0.1)
                pass
        return tokens_, masked_token, masked_idx, num_masked

    def mask_word(self, tokens, candidate_masked_idx):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        """
        token_ = copy.copy(tokens)
        masked_token = [token_[i] for i in candidate_masked_idx]
        for pos in candidate_masked_idx:
            token_[pos] = self.tokenizer.mask_token
        return token_, masked_token

    def pad_img_feat(self, img_feat, max_img_seq_len, img_dim):
        img_len = img_feat.size(0)
        if img_len > max_img_seq_len:
            img_feat = img_feat[0: max_img_seq_len, ]
            img_len = img_feat.size(0)
        else:
            padding_matrix = torch.zeros((max_img_seq_len - img_len, img_dim), device=self.device)
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        return img_feat, img_len

    def tensorize_example(self, text_a, turn_img_feat, ent_img_feat, text_b=None, text_c=None, text_e=None, cross_attn=True,
                          bos_token_segment_id=0, cls_token_segment_id=0, pad_token_segment_id=0, eos_token_segment_id=0,
                          sequence_a_segment_id=1, sequence_b_segment_id=2, sequence_c_segment_id=3, sequence_e_segment_id=4,
                          no_turn_vis=False, no_ent_vis=False, text_k=None):
        """
        convert data example to torch tensor

        :param text_a: response
        :param img_feat: 1xfeatsize + 1 + nxfeatsize: [turn-level image features] 'SEP' [entity level images]
        :param text_b: context
        :param text_c: captions
        :param text_e: entities
        :param cls_token_segment_id:
        :param pad_token_segment_id:
        :param sequence_a_segment_id:
        :param sequence_b_segment_id:
        :param sequence_e_segment_id:
        :param no_turn_vis: no turn-level visual knowledge
        :param no_ent_vis: no entity-level visual knowledge
        :param text_k: knowledge text
        :return:
        For training:

        input_ids is like:
        [BOS] Dialog Context (with [MASK]) [CONSEP] Caption (with [MASK]) [CAPSEP] Entities [EOS] [PAD]...[PAD]

        input_ids_rmask is like:
        [BOS] [MASK]...[MASK] [SEP] .. [PAD] [CLS] Dialog Context [CONSEP] Caption [CAPSEP] Entities [EOS] [PAD]...[PAD]
        """
        ## 1 for bos_token, 1 for sep_token
        ## tokenize withour padding: make sure that every sentence
        ## begins with bos_token and ends with eos_token
        EXTRA_A_TOKEN_LEN = 2
        tokens_a = self.tokenizer.tokenize(text_a)
        ## Use EDA for text data augmentation
        text_a_eda, tokens_a_eda, tokens_a_eda_len = None, None, None
        if self.res_eda:
            text_a_eda = copy.copy(text_a)
            text_a_eda = eda_proc(text_a_eda, num_aug=self.num_aug)[0]
            tokens_a_eda = self.tokenizer.tokenize(text_a_eda)
            tokens_a_eda_len = len(tokens_a_eda)

        if len(tokens_a) > self.max_seq_a_len - EXTRA_A_TOKEN_LEN:
            tokens_a = tokens_a[:(self.max_seq_a_len - EXTRA_A_TOKEN_LEN)]

        if tokens_a_eda_len is not None:
            if tokens_a_eda_len < self.max_seq_a_len:
                a_padding_len = self.max_seq_a_len - tokens_a_eda_len
                tokens_a_eda += ([self.tokenizer.pad_token] * a_padding_len)
            else:
                tokens_a_eda = tokens_a_eda[: self.max_seq_a_len]
                a_end = len(tokens_a)

        tokens_a = [self.tokenizer.bos_token] + tokens_a + [self.tokenizer.eos_token]
        seq_a_len = len(tokens_a)
        padding_a_len = self.max_seq_a_len - seq_a_len
        tokens_a += [self.tokenizer.pad_token] * padding_a_len

        assert text_b is not None, "Dialog context should not be None.."
        ## process text_b
        EXTRA_TOKEN_LEN = 2 ## extra special tokens of context (text_b)
        cap_seq_len = 0 if text_c is None else (self.max_turn_img_seq_len * self.per_max_turn_img_seq_len)
        ent_seq_len = 0 if text_e is None else self.max_ent_img_seq_len
        max_b_seq_len = self.max_seq_len - cap_seq_len - ent_seq_len - EXTRA_TOKEN_LEN
        max_b_end = max_b_seq_len
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_b_len = len(tokens_b)

        ## Use EDA for text data augmentation
        text_b_eda, tokens_b_eda, tokens_b_eda_len = None, None, None
        if self.conx_eda:
            text_b_eda = copy.copy(text_b)
            text_b_eda = eda_proc(text_b_eda, num_aug=self.num_aug)[0]
            tokens_b_eda = self.tokenizer.tokenize(text_b_eda)
            tokens_b_eda_len = len(tokens_b_eda)

        if tokens_b_len < max_b_seq_len:
            b_end = tokens_b_len
            b_padding_len = max_b_seq_len - tokens_b_len
            tokens_b += ([self.tokenizer.pad_token] * b_padding_len)
            segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * tokens_b_len + [pad_token_segment_id] * b_padding_len
        else:
            tokens_b = tokens_b[: max_b_seq_len]
            segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
            b_end = len(tokens_b)

        if tokens_b_eda_len is not None:
            if tokens_b_eda_len < max_b_seq_len:
                b_padding_len = max_b_seq_len - tokens_b_eda_len
                tokens_b_eda += ([self.tokenizer.pad_token] * b_padding_len)
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * tokens_b_eda_len + [
                    pad_token_segment_id] * b_padding_len
            else:
                tokens_b_eda = tokens_b_eda[: max_b_seq_len]
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
                b_end = len(tokens_b)



        end_of_seq_b = self.tokenizer.sep_token if not (no_ent_vis and no_turn_vis) else self.tokenizer.eos_token
        tokens = [self.tokenizer.bos_token] + tokens_b + [end_of_seq_b]
        tokens_eda = [self.tokenizer.bos_token] + tokens_b_eda + [end_of_seq_b] if tokens_b_eda is not None else None
        context_pos = list(range(1, b_end))
        segment_ids += [sequence_b_segment_id]

        ## lets not using captions for now (NOT updated)
        cap_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        if text_c is not None and not no_turn_vis:
            cap_idx = []
            ## process text_c captions
            c_start = len(tokens)
            c_end = c_start + self.max_turn_img_seq_len * self.per_max_turn_img_seq_len
            if len(text_c) < self.max_turn_img_seq_len:
                turn_img_seq_padding_len = self.max_turn_img_seq_len - len(text_c)
            else:
                text_c = text_c[: self.max_turn_img_seq_len]
                turn_img_seq_padding_len = 0
            c_start_tmp = c_start
            for turn_cap in text_c:
                tokens_c = self.tokenizer.tokenize(turn_cap)
                tokens_c_len = len(tokens_c)
                if tokens_c_len < self.per_max_turn_img_seq_len - 1: ## exclude caption sep special token
                    c_end_tmp = len(tokens) + tokens_c_len
                    c_padding_len = self.per_max_turn_img_seq_len - len(tokens_c) - 1
                    tokens_c += ([self.tokenizer.pad_token] * c_padding_len)
                    segment_ids += [sequence_c_segment_id] * tokens_c_len + [pad_token_segment_id] * c_padding_len
                else:
                    tokens_c = tokens_c[: self.per_max_turn_img_seq_len - 1]
                    segment_ids += [sequence_c_segment_id] * (self.per_max_turn_img_seq_len - 1)
                    c_end_tmp = len(tokens) + self.per_max_turn_img_seq_len - 1
                cap_idx.extend(list(range(c_start_tmp, c_end_tmp)))
                tokens += tokens_c + [self.tokenizer.sep_token]
                segment_ids += [sequence_c_segment_id]
                c_start_tmp = len(tokens)
            tokens += ([self.tokenizer.pad_token] * (self.per_max_turn_img_seq_len - 1) + [self.tokenizer.sep_token]) * turn_img_seq_padding_len
            segment_ids += ([pad_token_segment_id] * (self.per_max_turn_img_seq_len - 1) + [sequence_c_segment_id]) * turn_img_seq_padding_len
            cap_pos[cap_idx] = 1
        else:
            c_start = None
            c_end = None
            cap_idx = None

        if text_e is not None:
            ## process text_e entities
            e_start = len(tokens)
            if tokens_eda is not None:
                e_start_eda = len(tokens_eda)
            text_e = " ".join(text_e)
            tokens_e = self.tokenizer.tokenize(text_e)
            tokens_e_len = len(tokens_e)
            ## prepare for multimodal entity regression
            if tokens_e_len < self.max_ent_img_seq_len - 1: ## exclude eos special token
                e_padding_len = self.max_ent_img_seq_len - tokens_e_len - 1
                tokens_e += ([self.tokenizer.pad_token] * e_padding_len)
                segment_ids += [sequence_e_segment_id] * tokens_e_len + [pad_token_segment_id] * e_padding_len
            else:
                tokens_e = tokens_e[: self.max_ent_img_seq_len - 1]
                segment_ids += [sequence_e_segment_id] * (self.max_ent_img_seq_len - 1)

            tokens += tokens_e + [self.tokenizer.eos_token]
            if tokens_eda is not None:
                tokens_eda += tokens_e + [self.tokenizer.eos_token]
                e_end_eda = len(tokens_eda) - 1
                ent_pos_eda = list(range(e_start_eda, e_end_eda))
            ## because there may be no entity for dialog turn, e_end includes padding tokens
            e_end = len(tokens) - 1
            segment_ids += [eos_token_segment_id]
            ent_ids = self.tokenizer.convert_tokens_to_ids(tokens_e)
            ent_pos = list(range(e_start, e_end))
        else:
            e_start = e_end = 0
            ent_ids = None
            ent_pos = None
            ent_pos_eda = None
            # e_start = len(tokens)
            # tokens_e = [self.tokenizer.pad_token] * (self.max_ent_img_seq_len - 1) ## exclude eos token
            # tokens += tokens_e + [self.tokenizer.eos_token]
            # e_end = len(tokens) - 1
            # segment_ids += [pad_token_segment_id] * (self.max_ent_img_seq_len - 1) + [eos_token_segment_id]
            # ent_ids = self.tokenizer.convert_tokens_to_ids(tokens_e)

        if self.knowledge_len:
            ## process text_e entities
            k_start = len(tokens)
            tokens_k = self.tokenizer.tokenize(text_k)
            tokens_k_len = len(tokens_k)
            ## prepare for multimodal entity regression
            if tokens_k_len < self.knowledge_len - 1:  ## exclude eos special token
                k_padding_len = self.knowledge_len - tokens_k_len - 1
                tokens_k += ([self.tokenizer.pad_token] * k_padding_len)
                segment_ids += [sequence_b_segment_id] * tokens_k_len + [pad_token_segment_id] * k_padding_len
            else:
                tokens_k = tokens_k[: self.knowledge_len - 1]
                segment_ids += [sequence_b_segment_id] * (self.knowledge_len - 1)

            tokens += tokens_k + [self.tokenizer.eos_token]
            k_end = len(tokens) - 1
            segment_ids += [eos_token_segment_id]
            know_ids = self.tokenizer.convert_tokens_to_ids(tokens_k)
            know_pos = list(range(k_start, k_end))

        if self.is_train:
            masked_res_pos = torch.zeros(self.max_seq_a_len, dtype=torch.int)
            total_num_masked = 0
            masked_token_response = None
            # masked_token_tb = None

            if self.max_masked_response_tokens or self.mask_response_prob:
                ## mask response
                candidate_masked_response_idx = list(range(1, seq_a_len))
                masked_tokens_a, masked_token_response, masked_idx_response, num_masked_response = self.random_word(tokens_a,
                                                                                                             self.mask_response_prob, candidate_masked_response_idx,
                                                                                                             len(candidate_masked_response_idx), self.max_masked_response_tokens)
                masked_res_pos[masked_idx_response] = 1
                total_num_masked += num_masked_response

            assert self.max_masked_context_tokens == 0 and self.mask_prob == 0, "Not supporting masking context for now.."
            # if self.max_masked_context_tokens or self.mask_prob:
            #     # -----------------------------mask context part-----------------------------
            #     candidate_masked_idx_text_b = list(range(1, b_end)) ## bos token before text_b
            #     if cap_idx is not None:
            #         candidate_masked_idx_text_b.extend(cap_idx)
            #         # list(range(self.max_seq_a_len, tag_start))  # mask context
            #     ## mask context
            #     random.shuffle(candidate_masked_idx_text_b)
            #     tokens, masked_token_tb, masked_idx_tb, num_masked_tb = self.random_word(tokens, self.mask_prob, candidate_masked_idx_text_b,
            #                                                                              len(candidate_masked_idx_text_b), self.max_masked_context_tokens)
            #
            #     #  MRM with MLM in the same time
            #     masked_pos[masked_idx_tb] = 1
            #     # pad masked tokens to the same length
            #     total_num_masked += num_masked_tb

            if masked_token_response is not None:
                max_masked_tokens = self.max_masked_response_tokens #+ self.max_masked_context_tokens
                if total_num_masked < max_masked_tokens:
                    masked_token = masked_token_response + ([self.tokenizer.pad_token] *
                                                                     (max_masked_tokens - total_num_masked))
                else:
                    masked_token = masked_token_response

                masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
            else:
                masked_ids = None
        else:
            masked_ids = None
            masked_res_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids_eda = None
        ## only support context EDA or response EDA for now (NOT both)
        if tokens_eda is not None:
            input_ids_eda = self.tokenizer.convert_tokens_to_ids(tokens_eda)
        elif tokens_a_eda is not None:
            input_ids_eda = self.tokenizer.convert_tokens_to_ids(tokens_a_eda)

        decoder_input_ids = self.tokenizer.convert_tokens_to_ids(tokens_a)
        if self.is_train:
            masked_decoder_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens_a)
        else:
            masked_decoder_input_ids = None

        # image features
        turn_img_feat, ent_img_feat = turn_img_feat.to(self.device), ent_img_feat.to(self.device)
        turn_img_feat, turn_img_len = self.pad_img_feat(turn_img_feat, self.max_turn_img_seq_len, self.img_dim)
        ent_img_feat, ent_img_len = self.pad_img_feat(ent_img_feat, self.max_ent_img_seq_len, self.img_dim)
        # img_feat = torch.cat([ent_img_feat, turn_img_feat], dim=0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        seq_len = len(tokens)
        img_seq_len = 0
        if not no_turn_vis:
            img_seq_len += self.max_turn_img_seq_len
        if not no_ent_vis:
            img_seq_len += self.max_ent_img_seq_len
        ent_img_pos = torch.zeros(img_seq_len, dtype=torch.int, device=self.device)

        img_prefix_length = seq_len if not cross_attn else 0
        turn_img_pos_list = list(range(img_prefix_length, img_prefix_length + turn_img_len)) if not no_turn_vis else None
        if not no_ent_vis:
            if not no_turn_vis:
                ent_img_pos_list = list(range(img_prefix_length + self.max_turn_img_seq_len,
                                              img_prefix_length + self.max_turn_img_seq_len + ent_img_len))
            else:
                ent_img_pos_list = list(range(img_prefix_length, img_prefix_length + ent_img_len))
            ent_img_pos[list(range(ent_img_len))] = 1
        else:
            ent_img_pos_list = None

        if not (no_ent_vis and no_turn_vis):
            ## seq2seq model accepts 1-D attention mask
            if not cross_attn:
                max_len = seq_len + img_seq_len
                # attention_mask = torch.tensor(input_ids!=self.tokenizer.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((max_len,), dtype=torch.long, device=self.device)
                vis_pos = []
                vis_pos.extend(context_pos)
                if ent_pos is not None:
                    vis_pos.extend(ent_pos)
                if turn_img_pos_list is not None:
                    vis_pos.extend(turn_img_pos_list)
                if ent_img_pos_list is not None:
                    vis_pos.extend(ent_img_pos_list)
                attention_mask[vis_pos] = 1
                attention_mask = [attention_mask]
            else:
                attention_mask = torch.zeros((seq_len, ), dtype=torch.long, device=self.device)
                vis_pos = []
                vis_pos.extend(context_pos)
                if ent_pos is not None:
                    vis_pos.extend(ent_pos)
                attention_mask[vis_pos] = 1

                img_vis_pos = []
                img_attention_mask = torch.zeros((img_seq_len, ), dtype=torch.long, device=self.device)
                if turn_img_pos_list is not None:
                    img_vis_pos.extend(turn_img_pos_list)
                if ent_img_pos_list is not None:
                    img_vis_pos.extend(ent_img_pos_list)
                img_attention_mask[img_vis_pos] = 1
                attention_mask = [attention_mask, img_attention_mask]
        else:
            ## no visual knowledge at all
            attention_mask = torch.zeros((seq_len,), dtype=torch.long, device=self.device)
            vis_pos = []
            vis_pos.extend(context_pos)
            attention_mask[vis_pos] = 1
            # attention_mask = torch.tensor(input_ids != self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = [attention_mask]

        decoder_attention_mask = torch.zeros((self.max_seq_a_len,), dtype=torch.long, device=self.device)
        decoder_attention_mask[ :seq_a_len] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_ids_eda = torch.tensor(input_ids_eda, dtype=torch.long, device=self.device) if input_ids_eda is not None else 0
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=self.device)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long, device=self.device)
        if self.is_train:
            masked_decoder_input_ids = torch.tensor(masked_decoder_input_ids, dtype=torch.long, device=self.device)

        # used for get entity position
        ent_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_e:
        ent_idx = list(range(e_start, e_end))
        ent_pos[ent_idx] = 1

        # cap_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_c:
        #     cap_idx = list(range(c_start, c_end))
        #     cap_pos[cap_idx] = 1

        if self.is_train:
            assert masked_ids is not None
            masked_ids = torch.tensor(masked_ids, dtype=torch.long, device=self.device)
            ent_ids = torch.tensor(ent_ids, dtype=torch.long, device=self.device) if ent_ids is not None else None
            # masked_ent_ids = torch.tensor(masked_ent_ids, dtype=torch.long, device=self.device)
            if ent_ids is not None:
                return (decoder_input_ids, masked_decoder_input_ids, input_ids, decoder_attention_mask,
                    attention_mask, segment_ids, turn_img_feat, ent_img_feat, masked_res_pos,
                    masked_ids, cap_pos, ent_pos, ent_img_pos, ent_ids, input_ids_eda)
            else:
                return (decoder_input_ids, masked_decoder_input_ids, input_ids, decoder_attention_mask,
                    attention_mask, segment_ids, turn_img_feat, ent_img_feat, masked_res_pos,
                    masked_ids, cap_pos, ent_pos, ent_img_pos, input_ids_eda)
        else:
            return (decoder_input_ids, input_ids, decoder_attention_mask, attention_mask, segment_ids,
                    turn_img_feat, ent_img_feat, cap_pos, ent_pos, ent_img_pos)

class MMTensorizerGPT(object):
    def __init__(self, tokenizer, device, img_dim, max_turn_img_seq_length=5, per_max_turn_img_seq_length=20,
                 max_ent_img_seq_length=8, max_seq_a_length=30, max_seq_len=100, mask_prob=0, mask_response_prob=0.5,
                 max_masked_response_tokens=10, max_masked_context_tokens=0, random_token_list=None, is_train=True,
                 EDA=True, num_aug=9, knowledge_len=0):
        """ Dataset Constructor.
        :param tokenizer: text tokenizer
        :param device:
        :param max_turn_img_seq_length: max turn level image number
        :param max_ent_img_seq_length: max entity number / entity level image number
        :param max_seq_a_length: max response sequence length
        :param max_seq_len: max total textual sequence length
        :param mask_prob: mask prob for
        :param mask_response_prob: response mask prob
        :param max_masked_tokens:
        :param context_sep: tokenizer special token
        :param caption_sep: tokenizer special token
        :param num_special_tokens:
        :param is_train:
        """
        self.tokenizer = tokenizer
        self.device = device
        self.is_train = is_train
        self.eda = EDA
        self.num_aug = num_aug
        self.knowledge_len = knowledge_len
        self.max_turn_img_seq_len = max_turn_img_seq_length
        self.per_max_turn_img_seq_len = per_max_turn_img_seq_length
        self.max_ent_img_seq_len = max_ent_img_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_response_prob = mask_response_prob
        self.max_masked_response_tokens = max_masked_response_tokens
        self.max_masked_context_tokens = max_masked_context_tokens
        self.random_token_list = random_token_list
        self.img_dim = img_dim

    def pad_img_feat(self, img_feat, max_img_seq_len, img_dim):
        img_len = img_feat.size(0)
        if img_len > max_img_seq_len:
            img_feat = img_feat[0: max_img_seq_len, ]
            img_len = img_feat.size(0)
        else:
            padding_matrix = torch.zeros((max_img_seq_len - img_len, img_dim), device=self.device)
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        return img_feat, img_len

    def tensorize_example(self, text_a, turn_img_feat, ent_img_feat, text_b=None, text_c=None, text_e=None, cross_attn=True,
                          bos_token_segment_id=0, cls_token_segment_id=0, pad_token_segment_id=0, eos_token_segment_id=0,
                          sequence_a_segment_id=1, sequence_b_segment_id=2, sequence_c_segment_id=3, sequence_e_segment_id=4,
                          no_turn_vis=False, no_ent_vis=False, text_k=None):
        """
        convert data example to torch tensor
        for multi modal gpt model, the information is organized as for `concat`:
        [visual knowledge] [entity] [Dialog Context] [Response]

        for `cross_attn`:
        [entity] [Dialog Context] [Response]
        cross attention input: [visual knowledge]

        :param text_a: response
        :param img_feat: 1xfeatsize + 1 + nxfeatsize: [turn-level image features] 'SEP' [entity level images]
        :param text_b: context
        :param text_c: captions
        :param text_e: entities
        :param cls_token_segment_id:
        :param pad_token_segment_id:
        :param sequence_a_segment_id:
        :param sequence_b_segment_id:
        :param sequence_e_segment_id:
        :param no_turn_vis: no turn-level visual knowledge
        :param no_ent_vis: no entity-level visual knowledge
        :param text_k: knowledge text
        :return:
        """
        ## 1 for bos_token, 1 for sep_token
        ## tokenize withour padding: make sure that every sentence
        ## begins with bos_token and ends with eos_token
        # EXTRA_A_TOKEN_LEN = 2
        # tokens_a = self.tokenizer.tokenize(text_a)
        # if len(tokens_a) > self.max_seq_a_len - EXTRA_A_TOKEN_LEN:
        #     tokens_a = tokens_a[:(self.max_seq_a_len - EXTRA_A_TOKEN_LEN)]
        # tokens_a = [self.tokenizer.eos_token] + tokens_a + [self.tokenizer.eos_token]
        # seq_a_len = len(tokens_a)
        # padding_a_len = self.max_seq_a_len - seq_a_len
        # tokens_a += [self.tokenizer.pad_token] * padding_a_len

        ## place entity at the beginning for response generation
        if text_e is not None:
            ## process text_e entities
            e_start = 1
            e_start_eda = 1
            text_e = " ".join(text_e)
            tokens_e = self.tokenizer.tokenize(text_e)
            tokens_e_len = len(tokens_e)
            ## prepare for multimodal entity regression
            if tokens_e_len < self.max_ent_img_seq_len - 1: ## exclude eos special token
                e_padding_len = self.max_ent_img_seq_len - tokens_e_len - 1
                tokens_e += ([self.tokenizer.pad_token] * e_padding_len)
                segment_ids = [bos_token_segment_id] + [sequence_e_segment_id] * tokens_e_len + [pad_token_segment_id] * e_padding_len
            else:
                tokens_e = tokens_e[: self.max_ent_img_seq_len - 1]
                segment_ids = [bos_token_segment_id] + [sequence_e_segment_id] * (self.max_ent_img_seq_len - 1)

            tokens = [self.tokenizer.bos_token] + tokens_e
            tokens_eda = [self.tokenizer.bos_token] + tokens_e if self.eda else None
            ## because there may be no entity for dialog turn, e_end includes padding tokens
            e_end = len(tokens)
            ent_ids = self.tokenizer.convert_tokens_to_ids(tokens_e)
            ent_pos = list(range(e_start, e_end))
        else:
            e_start = e_end = 0
            ent_ids = None
            ent_pos = None
            ent_pos_eda = None

        assert text_b is not None, "Dialog context should not be None.."
        ## process text_b
        EXTRA_TOKEN_LEN = 2 ## extra special tokens of context (text_b)
        ent_seq_len = 0 if text_e is None else self.max_ent_img_seq_len
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_b_len = len(tokens_b)
        max_b_seq_len = self.max_seq_len- ent_seq_len - EXTRA_TOKEN_LEN

        b_start = 1 if text_e is None else len(tokens) + 1
                ## Use EDA for text data augmentation
        text_b_eda, tokens_b_eda, tokens_b_eda_len = None, None, None
        # if self.is_train:
        if self.eda:
            text_b_eda = copy.copy(text_b)
            text_b_eda = eda_proc(text_b_eda, num_aug=self.num_aug)[0]
            tokens_b_eda = self.tokenizer.tokenize(text_b_eda)
            tokens_b_eda_len = len(tokens_b_eda)

        if tokens_b_len < max_b_seq_len:
            b_end = tokens_b_len + b_start - 1
            b_padding_len = max_b_seq_len - tokens_b_len
            tokens_b += ([self.tokenizer.pad_token] * b_padding_len)
            if text_e is None:
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * tokens_b_len + [pad_token_segment_id] * b_padding_len
            else:
                segment_ids += [eos_token_segment_id] + [sequence_b_segment_id] * tokens_b_len + [pad_token_segment_id] * b_padding_len
        else:
            tokens_b = tokens_b[: max_b_seq_len]
            if text_e is None:
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
            else:
                segment_ids += [eos_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
            b_end = len(tokens_b) + b_start - 1

        ## EDA process for dialogue context
        if tokens_b_eda_len is not None:
            if tokens_b_eda_len < max_b_seq_len:
                b_padding_len = max_b_seq_len - tokens_b_eda_len
                tokens_b_eda += ([self.tokenizer.pad_token] * b_padding_len)
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * tokens_b_eda_len + [
                    pad_token_segment_id] * b_padding_len
            else:
                tokens_b_eda = tokens_b_eda[: max_b_seq_len]
                segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * max_b_seq_len
                b_end = len(tokens_b)
        # else:
        #     ## When testing, we only consider batch size = 1
        #     ## to make sure that every context is with full
        #     ## length for answering
        #     b_end = tokens_b_len + b_start - 1
        #     if text_e is None:
        #         segment_ids = [bos_token_segment_id] + [sequence_b_segment_id] * tokens_b_len
        #     else:
        #         segment_ids += [eos_token_segment_id] + [sequence_b_segment_id] * tokens_b_len


        if text_e is None:
            tokens = [self.tokenizer.bos_token] + tokens_b + [self.tokenizer.eos_token]
            tokens_eda = [self.tokenizer.bos_token] + tokens_b_eda + [self.tokenizer.eos_token] if self.eda else None
        else:
            tokens += [self.tokenizer.eos_token] + tokens_b + [self.tokenizer.eos_token]
            if self.eda and self.is_train:
                tokens_eda += [self.tokenizer.eos_token] + tokens_b_eda + [self.tokenizer.eos_token]
        context_pos = list(range(b_start, b_end))
        segment_ids += [eos_token_segment_id]

        ## Lets not support caption as input for now
        if text_c is not None:
            pass

        if self.knowledge_len:
            ## process text_e entities
            k_start = len(tokens)
            tokens_k = self.tokenizer.tokenize(text_k)
            tokens_k_len = len(tokens_k)
            ## prepare for multimodal entity regression
            if tokens_k_len < self.knowledge_len - 1:  ## exclude eos special token
                k_padding_len = self.knowledge_len - tokens_k_len - 1
                tokens_k += ([self.tokenizer.pad_token] * k_padding_len)
                segment_ids += [sequence_b_segment_id] * tokens_k_len + [pad_token_segment_id] * k_padding_len
            else:
                tokens_k = tokens_k[: self.knowledge_len - 1]
                segment_ids += [sequence_b_segment_id] * (self.knowledge_len - 1)

            tokens += tokens_k + [self.tokenizer.eos_token]
            k_end = len(tokens) - 1
            segment_ids += [eos_token_segment_id]
            know_ids = self.tokenizer.convert_tokens_to_ids(tokens_k)
            know_pos = list(range(k_start, k_end))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids_eda = self.tokenizer.convert_tokens_to_ids(tokens_eda) if tokens_eda is not None else None

        # image features
        turn_img_feat, ent_img_feat = turn_img_feat.to(self.device), ent_img_feat.to(self.device)
        turn_img_feat, turn_img_len = self.pad_img_feat(turn_img_feat, self.max_turn_img_seq_len, self.img_dim)
        ent_img_feat, ent_img_len = self.pad_img_feat(ent_img_feat, self.max_ent_img_seq_len, self.img_dim)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        seq_len = len(tokens)
        img_seq_len = 0

        ## the image is designed to be at the beginning of a dialogue sequence
        if not no_turn_vis:
            img_seq_len += self.max_turn_img_seq_len
            if not cross_attn:
                context_pos = [item + self.max_turn_img_seq_len for item in context_pos]
        if not no_ent_vis:
            img_seq_len += self.max_ent_img_seq_len
            if not cross_attn:
                context_pos = [item + self.max_ent_img_seq_len for item in context_pos]
        ent_img_pos = torch.zeros(img_seq_len, dtype=torch.int, device=self.device)

        turn_img_pos_list = list(range(turn_img_len)) if not no_turn_vis else None
        if not no_ent_vis:
            if not no_turn_vis:
                ent_img_pos_list = list(range(self.max_turn_img_seq_len, self.max_turn_img_seq_len + ent_img_len))
            else:
                ent_img_pos_list = list(range(ent_img_len))
            ent_img_pos[list(range(ent_img_len))] = 1
        else:
            ent_img_pos_list = None

        if not (no_ent_vis and no_turn_vis):
            ## seq2seq model accepts 1-D attention mask
            if not cross_attn:
                max_len = seq_len + img_seq_len
                ## use segment embedding (c segment id) to distinguish visual, textual knowledge
                segment_ids = [sequence_c_segment_id] * img_seq_len + segment_ids
                # attention_mask = torch.tensor(input_ids!=self.tokenizer.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((max_len,), dtype=torch.long, device=self.device)
                vis_pos = []
                vis_pos.extend(context_pos)
                if ent_pos is not None:
                    vis_pos.extend(ent_pos)
                if turn_img_pos_list is not None:
                    vis_pos.extend(turn_img_pos_list)
                if ent_img_pos_list is not None:
                    vis_pos.extend(ent_img_pos_list)
                attention_mask[vis_pos] = 1
                attention_mask = [attention_mask]
            else:
                max_len = seq_len
                # attention_mask = torch.tensor(input_ids!=self.tokenizer.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((max_len,), dtype=torch.long, device=self.device)
                vis_pos = []
                vis_pos.extend(context_pos)
                attention_mask[vis_pos] = 1
                attention_mask = [attention_mask]
        else:
            ## no visual knowledge at all
            attention_mask = torch.zeros((seq_len,), dtype=torch.long, device=self.device)
            vis_pos = []
            vis_pos.extend(context_pos)
            attention_mask[vis_pos] = 1
            # attention_mask = torch.tensor(input_ids != self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = [attention_mask]

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_ids_eda = torch.tensor(input_ids_eda, dtype=torch.long, device=self.device) if input_ids_eda is not None else 0
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=self.device)

        # used for get entity position
        ent_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_e:
        ent_idx = list(range(e_start, e_end))
        ent_pos[ent_idx] = 1

        # cap_pos = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        # if text_c:
        #     cap_idx = list(range(c_start, c_end))
        #     cap_pos[cap_idx] = 1
        ent_ids = torch.tensor(ent_ids, dtype=torch.long, device=self.device) if ent_ids is not None else 0
        if self.is_train:
            # masked_ent_ids = torch.tensor(masked_ent_ids, dtype=torch.long, device=self.device)
            return (input_ids, attention_mask, segment_ids, turn_img_feat, ent_img_feat,
                    ent_pos, ent_img_pos, input_ids_eda, ent_ids)
        else:
            return (input_ids, attention_mask, segment_ids, turn_img_feat, ent_img_feat,
                    ent_pos, ent_img_pos, ent_ids)
