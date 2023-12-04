#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@file: run.py
@author: ImKe at 2022/6/11
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import datetime, math, os, sys, json, argparse, time, re, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tensorboardX import SummaryWriter
sys.path.append("./")
from logger import Logger
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    BertTokenizer,
    BertConfig,
    T5Tokenizer,
    T5Config,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup)
from torch.optim import Adam
from tqdm import tqdm
from utils import model_utils
from model.base import GPT2ModelForMultiModal, GPT2LMforMMDialog
from model.base_bert import BertForMMDialog
from model.base_t5 import T5ForMMDialogGeneration
from data_utils.dialog_reader import MMTensorizer, MMDialogDataset
from utils.eval_utils import evaluate_on_wow_multimodal_training

configs = json.load(open("config.json"))

cache_dir = configs['CACHE_DIR']
data_dir = configs['DATA_DIR']
MIN_PPL_THR=10
FIXED_GPT_LEN=250

# episodes, entity_turn, caption_turn, entities_img_feats, turn_img_feats, history_in_context, max_episode_length=1
parser = argparse.ArgumentParser()
## data preparation
parser.add_argument("--dataset", default='WOW', type=str, choices=["WOW", "DD"],
                    help="Training dataset. WOW for Wizard of Wikipedia, DD for DailyDialog")

parser.add_argument("--episodes_train_file", default='./wow/train.json', type=str)
parser.add_argument("--episodes_val_file", default='./wow/valid_topic_split.json', type=str)
parser.add_argument("--episodes_test_file", default='./wow/test_topic_split.json', type=str)
parser.add_argument("--trainloader_file", default='./wow/processed_data/IP-1turn.json', type=str)
parser.add_argument("--valloader_file", default='./wow/processed_data/IP-valid_topic-1turn.json', type=str)
parser.add_argument("--testloader_file", default='./wow/processed_data/IP-test_topic-1turn.json', type=str)

parser.add_argument("--model_type", default="t5", type=str, choices=['gpt', 't5', 'unilm'])
parser.add_argument("--unilm_cache", default='/usr/.cache/unilm-base', type=str)
parser.add_argument("--t5_model_name", default='t5-base', type=str)
parser.add_argument("--gpt_model_name", default='gpt2-medium',
                    type=str, choices=['gpt2-medium', 'microsoft/DialoGPT-medium'])
parser.add_argument("--load", default=None, type=str)

parser.add_argument("--history_in_context", action='store_true', help="Use one dialog session as multiple training sample")
parser.add_argument("--max_episode_length", default=50, type=int)
parser.add_argument("--output_dir", default='output/', type=str,
                    help="The output directory to save checkpoint and test results.")
parser.add_argument("--max_seq_length", default=185, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, "
                         "sequences shorter will be padded.")
parser.add_argument("--max_seq_a_length", default=30, type=int,
                    help="The maximum sequence length for caption.")
parser.add_argument("--knowledge_len", default=0, type=int,
                    help="The maximum sequence length for knowledge document for Wizard of Wikipedia dataset.")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_sample", action='store_true', help="Whether to do sampling.")
parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
parser.add_argument("--mask_prob", default=0.0, type=float,
                    help= "Probability to mask input sentence during training.")
parser.add_argument("--mask_response_prob", default=0.8, type=float,
                    help= "Probability to mask input sentence during training.")
parser.add_argument("--max_masked_response_tokens", type=int, default=30,
                    help="The max number of masked tokens of response.")
parser.add_argument("--max_masked_context_tokens", type=int, default=0,
                    help="The max number of masked tokens of context and caption.")
parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in the model.")

parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing")
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--beta", default=1.0, type=float)

parser.add_argument("--max_turn_img_length", default=5, type=int,
                    help="The maximum image number turn level image.")
parser.add_argument("--per_max_turn_img_seq_length", default=10, type=int,
                    help="The maximum image number turn level image.")
parser.add_argument("--max_ent_img_seq_length", default=8, type=int,
                    help="The maximum total input entity level image sequence length.")
parser.add_argument("--img_feature_dim", default=512, type=int,
                    help="The Image Feature Dimension default is 512 for CLIP-base. 1024 for CLIP-large")
parser.add_argument("--type_vocab_size", default=3, type=int,
                    help="segment embedding vocabulary size.")
parser.add_argument("--img_feature_type", default='clip', type=str,
                    help="Image feature type.")
parser.add_argument("--img_layer_norm_eps", default=1e-4, type=float,
                    help="segment embedding vocabulary size.")
parser.add_argument("--img_add_pos", default="concat", type=str,
                    help="image adding method",
                    choices=['concat', 'enc_cross_attn', 'dec_cross_attn', 'cross_attn'])
parser.add_argument("--use_img_layernorm", action='store_true')

parser.add_argument('--no_gpu', action='store_true')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument("--per_gpu_train_batch_size", default=42, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=5, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--workers", default=0, type=int,
                    help="Dataloader worker.")
parser.add_argument("--early_stop", default=5, type=int,
                    help="Early stopping iteration.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before backward.")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate.")
parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warm_up_ratio", default=0.1, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup.")
parser.add_argument("--scheduler", default='linear', type=str, help="linear or consine", choices=["linear", "consine"])
parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
parser.add_argument("--num_train_epochs", default=None, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--vit_version", default="base", type=str, choices=["base", "large"])
parser.add_argument("--iterations", default=30000, type=int,
                    help="Total number of training iterations to perform.")
parser.add_argument("--log_epoch", default=2, type=int)
parser.add_argument("--test_iter", default=10, type=int)
parser.add_argument("--save_per_test_iter", default=0, type=int)
parser.add_argument("--max_val_batches", default=200, type=int)
parser.add_argument("--max_test_batches", default=200, type=int)

parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Run evaluation during training at each save_steps.")
parser.add_argument("--evaluate_testset_during_training", action='store_true',
                    help="Run evaluation testset during training at each save_steps.")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization.")
parser.add_argument('--ent_img_num', type=int, default=1)
parser.add_argument('--add_response_mlm', action='store_true', help='Add masked response prediction')
parser.add_argument('--add_cap_bias', action='store_true', help='Add caption bias to logits')
parser.add_argument('--add_ent_bias', action='store_true', help='Add entity bias to logits')
parser.add_argument('--add_ent_matching', action='store_true')
parser.add_argument('--add_textual_cap', action='store_true',
                    help="Add captions if available for training")
parser.add_argument('--add_textual_ent', action='store_true',
                    help="Add entity texts if available for training")
parser.add_argument('--t5_mlm', action='store_true')
parser.add_argument('--eda', action='store_true', help='Use easy data augmentation for encoder input')
parser.add_argument('--eda_pos', type=str, default='res', choices=['res', 'conx'],
                    help='eda position for T5 based model, context or response')
parser.add_argument('--no_turn_vis', action='store_true',
                    help='Train the model without Turn-level Visual Knowledge')
parser.add_argument('--no_ent_vis', action='store_true',
                    help='Train the model without Entity-level Visual Knowledge')
parser.add_argument('--last_res', action='store_true',
                    help='Train the model with only the last response of dialogue data')
parser.add_argument('--bleu_eval', action='store_true',
                    help='Use BLEU-1 as the evaluation metric')


# for generation
parser.add_argument("--eval_model_dir", type=str, default="test_output",
                    help="Model directory for evaluation.")
parser.add_argument("--evaluate_cache", type=str, default="",
                    help="Model directory storing ckpt for model testing and evaluation.")
parser.add_argument('--max_gen_length', type=int, default=40,
                    help="max length of generated sentences")
parser.add_argument('--output_hidden_states', action='store_true',
                    help="Turn on for fast decoding")
parser.add_argument('--num_return_sequences', type=int, default=1,
                    help="repeating times per image")
parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
parser.add_argument('--num_keep_best', type=int, default=1,
                    help="number of hypotheses to keep in beam search")
parser.add_argument('--temperature', type=float, default=1,
                    help="temperature in softmax for sampling")
parser.add_argument('--top_k', type=int, default=10,
                    help="filter distribution for sampling")
parser.add_argument('--top_p', type=float, default=0.10,
                    help="filter distribution for sampling")
parser.add_argument('--repetition_penalty', type=float, default=3.0,
                    help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
parser.add_argument('--length_penalty', type=float, default=0.3,
                    help="beam search length penalty")
parser.add_argument('--add_cap_labels', action='store_true',
                    help="Add captions if available for generation")
parser.add_argument('--add_ent_labels', action='store_true',
                    help="Add entity texts if available for generation")
parser.add_argument('--dp', action='store_true')
# for Constrained Beam Search
parser.add_argument('--use_cbs', action='store_true',
                    help='Use constrained beam search for decoding')
parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                    help="minimum number of constraints to satisfy")

def test(args, test_dataloader, gen_dataloader, model, tokenizer, predict_file,
         logging, max_test_batches, generate, device):
    model.eval()
    test_step = min(len(test_dataloader), max_test_batches)
    gen_step = min(len(gen_dataloader), max_test_batches)

    with torch.no_grad():
        val_losses, val_masked_token_losses, val_masked_res_losses, \
        val_ent_matching_losses = [], [], [], []
        for step, data_dict in tqdm(enumerate(test_dataloader),
                                ncols=100, desc="Testing Forward: ",
                                total=test_step):
            img_attention_mask = data_dict["attention_mask"][1] if args.img_add_pos == "cross_attn" else None

            ## remove padding tokens
            masked_ids = data_dict["masked_ids"]
            masked_ids = masked_ids[masked_ids != tokenizer.pad_token_id]

            if args.add_response_mlm:
                masked_res = data_dict["masked_res"][:, 1:]  # remove BOS id at the beginning
                masked_res = masked_res[masked_res != tokenizer.pad_token_id]
                input_ids_rmask = data_dict["input_ids_rmask"]
            else:
                masked_res = None
                input_ids_rmask = None

            inputs = {"input_ids": data_dict["input_ids"],
                      "input_ids_rmask": input_ids_rmask,
                      "turn_img_feats": data_dict["turn_img_feats"],
                      "ent_img_feats": data_dict["ent_img_feats"],
                      "attention_mask": data_dict["attention_mask"][0],
                      "img_attention_mask": img_attention_mask,
                      "token_type_ids": data_dict["segment_ids"],
                      "masked_ids": masked_ids,
                      "masked_pos": data_dict["masked_pos"],
                      "masked_res": masked_res,
                      "masked_res_pos": data_dict["masked_res_pos"],
                      "add_ent_bias": args.add_ent_bias,
                      "add_cap_bias": args.add_cap_bias,
                      "cap_pos": data_dict["cap_pos"],
                      "ent_pos": data_dict["ent_pos"],
                      "add_ent_matching": args.add_ent_matching,
                      "ent_img_pos": data_dict["ent_img_pos"],
                      "ent_ids": data_dict["ent_ids"] if (not args.no_ent_vis and args.add_textual_ent) else None,
                      }
            model_output = model(**inputs)
            masked_token_loss, masked_res_loss, ent_matching_loss, class_logits, batch_masked_score, \
            batch_masked_res_score, batch_ent_score = model_output[:7]
            loss = args.alpha * masked_token_loss + (1 - args.alpha) * masked_res_loss + args.beta * ent_matching_loss

            val_losses.append(loss.item())
            val_masked_token_losses.append(masked_token_loss.item())
            val_masked_res_losses.append(masked_res_loss.item())
            val_ent_matching_losses.append(ent_matching_loss.item())

    def gen_rows():
        time_meter = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(gen_dataloader),
                                    ncols=100, desc="Testing Forward: ",
                                    total=gen_step):

                # batch = tuple(t.to(device) for t in batch)
                img_attention_mask = batch["attention_mask"][1] if args.img_add_pos == "cross_attn" else None

                inputs = {'is_decode': True,
                          'input_ids': batch['input_ids'],
                          'attention_mask': batch['attention_mask'][0],
                          'token_type_ids': batch['segment_ids'],
                          'masked_pos': batch['masked_pos'],
                          'turn_img_feats': batch['turn_img_feats'],
                          'ent_img_feats': batch['ent_img_feats'],
                          'img_attention_mask': img_attention_mask,

                          'gth_ids': batch['gth_ids'],
                          'do_sample': args.do_sample,
                          'bos_token_id': tokenizer.bos_token_id,
                          'pad_token_id': tokenizer.pad_token_id,
                          'eos_token_ids': tokenizer.eos_token_id,
                          'mask_token_id': tokenizer.mask_token_id,
                          # for adding entity labels
                          'add_ent_labels': args.add_ent_labels,
                          'add_cap_labels': args.add_cap_labels,
                          ## since we separately pad response, context, caption, entity tokens
                          ## their starting positions are the same across the same batch
                          'ent_labels_start_posid': batch['ent_labels_start_posid'][0],
                          'cap_start_posid': batch['cap_start_posid'][0],
                          'context_start_posid': batch['context_start_posid'][0],
                          'context_end_posid': batch['context_end_posid'][0],

                          # hyperparameters of beam search
                          'max_length': args.max_seq_a_length,
                          'num_beams': args.num_beams,
                          "temperature": args.temperature,
                          "top_k": args.top_k,
                          "top_p": args.top_p,
                          "repetition_penalty": args.repetition_penalty,
                          "length_penalty": args.length_penalty,
                          "num_return_sequences": args.num_return_sequences,
                          "num_keep_best": args.num_keep_best,
                          }
                if args.use_cbs:
                    inputs.update({'use_cbs': True,
                                   'fsm': batch[5],
                                   'num_constraints': batch[6],
                                   'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
                                   })
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # [batch_size, num_keep_best, max_len]
                all_confs = torch.exp(outputs[1])
                all_responses = batch["response"]

                for caps, confs, response in zip(all_caps, all_confs, all_responses):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        cap = cap.split('.')[0]
                        res.append({'response_pred': cap, 'response': response, 'conf': conf.item()})

                    yield res
                    # return res
                if step >= test_step:
                    break
        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))
    if generate:
        model_utils.tsv_writer(gen_rows(), predict_file)

    logging.info("========================= Validate Status =========================")
    logging.info("Avg. Val Loss            : {:.4f}".format(np.mean(val_losses)))
    logging.info("Avg. Val Masked LM Loss  : {:.4f}".format(np.mean(val_masked_token_losses)))
    logging.info("Avg. Val Masked PPL      : {:.4f}".format(math.exp(min(np.mean(val_masked_token_losses), MIN_PPL_THR))))
    logging.info("Avg. Val Masked Res Loss : {:.4f}".format(np.mean(val_masked_res_losses)))
    logging.info("Avg. Val Ent Matching    : {:.4f}".format(np.mean(val_ent_matching_losses)))

    return np.mean(val_losses), np.mean(val_masked_token_losses), np.mean(val_masked_res_losses), np.mean(val_ent_matching_losses)

def train_UniLM(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    else:
        args.train_batch_size = args.per_gpu_train_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    loading = True if args.load is not None else False

    if args.no_ent_vis and args.no_turn_vis:
        ex_appended = f"novis_{now.month}.{now.day}"
    elif args.no_ent_vis:
        ex_appended = f"no_evis_{now.month}.{now.day}"
    elif args.no_turn_vis:
        ex_appended = f"no_tvis_{now.month}.{now.day}"
    else:
        ex_appended = f"{now.month}.{now.day}"

    if args.last_res:
        ex_appended = f"lastres_{ex_appended}"

    experiment = f"train_unilm_{args.dataset}_{args.img_add_pos}_e{args.num_train_epochs}" \
                 f"_res-mlm-{args.add_response_mlm}_Ebiad-{args.add_ent_bias}_Ematch-{args.add_ent_matching}" \
                 f"_lr{args.learning_rate}_seqlen{args.max_seq_length}_bs{args.per_gpu_train_batch_size}_" \
                 f"load{loading}_cap{int(args.add_textual_cap)}_ent{int(args.add_textual_ent)}" \
                 f"_ep{args.max_episode_length}_{ex_appended}"

    save_folder = os.path.join(args.output_dir, experiment)
    os.makedirs(os.path.join(save_folder, 'ckpt/model'), exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    # v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    logging_file = f"train_unilm_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    ## prepare model & resize tokenizer
    config = BertConfig.from_pretrained(args.unilm_cache)
    config = model_utils.sort_config(config, args)
    tokenizer = BertTokenizer.from_pretrained(args.unilm_cache)

    state = torch.load(os.path.join(args.unilm_cache, "pytorch_model.bin"))
    state = model_utils.rearrange_unilm_weights(state)
    model = BertForMMDialog(config, args.img_add_pos)
    model.load_state_dict(state, strict=False)
    print("Finish Loading Pre-trained UniLM Model Weight")
    del state

    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    # special_tokens_dict = {'bos_token': '[CLS]', 'eos_token': '[SEP]',
    #                        'additional_special_tokens': ['[CONSEP]', '[CAPSEP]']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # logging.info(f'We have added {num_added_toks} tokens to the model..')

    # model._resize_token_embeddings(len(tokenizer))
    # model.bert._resize_token_embeddings(len(tokenizer))
    # model.resize_token_embeddings(len(tokenizer))
    bert_embedding_weight = model.bert.embeddings.word_embeddings.weight
    # model.decoder = nn.Linear(bert_embedding_weight.size(1),
    #                           bert_embedding_weight.size(0), bias=False)
    model.vocab_size = bert_embedding_weight.size(0)
    model.build_smoothing_loss()

    if args.load is not None:
        print("Loading Trained Model Weight...")
        state = torch.load(os.path.join("output", args.load, "model_best_val.pt"))
        model.load_state_dict(state)
        del state

    model.to(device)
    model.train()

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # if len(args.gpu) > 1:
    #     model = nn.DataParallel(model)

    p_train_data = model_utils.jsonload(args.trainloader_file)[0]
    p_val_data = model_utils.jsonload(args.valloader_file)[0]
    if args.dataset == "WOW":
        dialog_episodes = json.load(open(args.episodes_train_file))
        val_episodes = json.load(open(args.episodes_val_file))
    elif args.dataset == "DD":
        dialog_episodes = model_utils.jsonload(args.episodes_train_file)
        val_episodes = model_utils.jsonload(args.episodes_val_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
    else:
        ent_clip_fea_file = None
    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])

    random_token_list = model_utils.gen_random_word_list(tokenizer, config.vocab_size)

    train_data = MMDialogDataset(dialog_episodes, device,
                                 p_train_data["entity_turn"],
                                 p_train_data["caption_turn"],
                                 p_train_data["entities_img_feats_ids"][0],
                                 p_train_data["turn_img_feats_ids"],
                                 turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                 img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                 history_in_context=args.history_in_context,
                                 max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                 max_turn_img_seq_length=args.max_turn_img_length,
                                 per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                 max_ent_img_seq_length=args.max_ent_img_seq_length,
                                 max_seq_length=args.max_seq_length,
                                 max_seq_a_length=args.max_seq_a_length, is_train=args.do_train,
                                 mask_prob=args.mask_prob, mask_response_prob=args.mask_response_prob,
                                 response_sep="[SEP_0]", context_sep="[SEP_1]",  caption_sep="[SEP_2]",
                                 random_token_list=random_token_list, test=args.last_res,
                                 max_masked_response_tokens=args.max_masked_response_tokens,
                                 max_masked_context_tokens=args.max_masked_context_tokens,
                                 dataset=args.dataset, add_cap=args.add_textual_cap,
                                 add_ent=args.add_textual_ent, no_ent_vis=args.no_ent_vis,
                                 no_turn_vis=args.no_turn_vis)
    val_gen_data = MMDialogDataset(val_episodes, device,
                                p_val_data["entity_turn"],
                                p_val_data["caption_turn"],
                                p_val_data["entities_img_feats_ids"][0],
                                p_val_data["turn_img_feats_ids"],
                                turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                history_in_context=args.history_in_context,
                                max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                max_turn_img_seq_length=args.max_turn_img_length,
                                per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                max_ent_img_seq_length=args.max_ent_img_seq_length,
                                max_seq_length=args.max_seq_length,
                                max_seq_a_length=args.max_seq_a_length, is_train=False,
                                response_sep="[SEP_0]", context_sep="[SEP_1]",  caption_sep="[SEP_2]",
                                test=True, dataset=args.dataset,
                                add_cap=False if not args.add_cap_labels else args.add_textual_cap,
                                add_ent=False if not args.add_ent_labels else args.add_textual_ent,
                                no_ent_vis=args.no_ent_vis,
                                no_turn_vis=args.no_turn_vis)
    val_data = MMDialogDataset(val_episodes, device,
                               p_val_data["entity_turn"],
                               p_val_data["caption_turn"],
                               p_val_data["entities_img_feats_ids"][0],
                               p_val_data["turn_img_feats_ids"],
                               turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                               img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                               history_in_context=args.history_in_context,
                               max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                               max_turn_img_seq_length=args.max_turn_img_length,
                               per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                               max_ent_img_seq_length=args.max_ent_img_seq_length,
                               max_seq_length=args.max_seq_length,
                               max_seq_a_length=args.max_seq_a_length, is_train=True,
                               response_sep="[SEP_0]", context_sep="[SEP_1]",  caption_sep="[SEP_2]",
                               random_token_list=random_token_list,
                               test=True, dataset=args.dataset,
                               add_cap=args.add_textual_cap,
                               add_ent=args.add_textual_ent,
                               no_ent_vis=args.no_ent_vis,
                               no_turn_vis=args.no_turn_vis)

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              pin_memory=False,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=True)

    val_sampler = SequentialSampler(val_data)
    val_gen_loader = DataLoader(val_gen_data,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)
    val_loader = DataLoader(val_data,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=True, weight_decay=args.weight_decay)
    if args.num_train_epochs is None:
        total_step = args.iterations
    else:
        total_step = len(train_loader) * args.num_train_epochs
        logging.info(f"Ignoring assigned iterations, total step is {total_step}..")
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    else:
        raise NotImplementedError

    logging.info("Begin training steps")
    max_val_batches = args.max_val_batches  # max num. of val batches
    logging.info("Total step: %d" % total_step)
    e = 0  # number of epoch
    num_iters = 0
    best_loss = 9e9
    et = 0
    eval_step = 0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    while (num_iters < total_step) and (et < args.early_stop):
        # Run epoch
        st = time.time()
        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n===========================================================================')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        ## record for every epoch losses
        losses, masked_token_losses, masked_res_losses, ent_matching_losses = [], [], [], []
        batch_masked_scores, batch_ent_scores, batch_masked_res_scores = [], [], []
        for i, data_dict in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img_attention_mask = data_dict["attention_mask"][1] if args.img_add_pos == "cross_attn" else None

            ## remove padding tokens
            masked_ids = data_dict["masked_ids"]
            masked_ids = masked_ids[masked_ids != tokenizer.pad_token_id]

            if args.add_response_mlm:
                masked_res = data_dict["masked_res"][:, 1:]  # remove BOS id at the beginning
                masked_res = masked_res[masked_res != tokenizer.pad_token_id]
                input_ids_rmask = data_dict["input_ids_rmask"]
            else:
                masked_res = None
                input_ids_rmask = None

            inputs = {"input_ids": data_dict["input_ids"],
                      "input_ids_rmask": input_ids_rmask,
                      "turn_img_feats": data_dict["turn_img_feats"],
                      "ent_img_feats": data_dict["ent_img_feats"],
                      "attention_mask": data_dict["attention_mask"][0],
                      "img_attention_mask": img_attention_mask,
                      "token_type_ids": data_dict["segment_ids"],
                      "masked_ids": masked_ids,
                      "masked_pos": data_dict["masked_pos"],
                      "masked_res": masked_res,
                      "masked_res_pos": data_dict["masked_res_pos"],
                      "add_ent_bias": args.add_ent_bias,
                      "add_cap_bias": args.add_cap_bias,
                      "cap_pos": data_dict["cap_pos"],
                      "ent_pos": data_dict["ent_pos"],
                      "add_ent_matching": args.add_ent_matching,
                      "ent_img_pos": data_dict["ent_img_pos"],
                      "ent_ids": data_dict["ent_ids"] if not args.no_ent_vis else None,
                      }
            with torch.cuda.amp.autocast():
                model_output = model(**inputs)
                masked_token_loss, masked_res_loss, ent_matching_loss, class_logits, batch_masked_score, \
                batch_masked_res_score, batch_ent_score = model_output[:7]
                loss = args.alpha * masked_token_loss + (1 - args.alpha) * masked_res_loss + args.beta * ent_matching_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # return loss, masked_token_loss, masked_res_token_loss, ent_matching_loss, \
            #        batch_masked_score, batch_masked_res_score, batch_ent_score
            # loss, masked_token_loss, masked_res_loss, ent_matching_loss, batch_masked_score, \
            # batch_masked_res_score, batch_ent_score = unilm_train_step(model, optimizer, scaler,
            #                                                            data_dict, tokenizer.pad_token_id, args)
            if args.gpu and args.n_gpu > 1:
                loss = loss.mean()
                masked_token_loss = masked_token_loss.mean()
                masked_res_loss = masked_res_loss.mean()
                ent_matching_loss = ent_matching_loss.mean()

            losses.append(loss.item())
            masked_res_losses.append(masked_res_loss.item())
            masked_token_losses.append(masked_token_loss.item())
            ent_matching_losses.append(ent_matching_loss.item())
            batch_masked_scores.append(batch_masked_score.item())
            batch_ent_scores.append(batch_ent_score.item())
            batch_masked_res_scores.append(batch_masked_res_score.item())
            lr = scheduler.get_last_lr()[0]

            # Log to Tensorboard
            t_writer.add_scalar('loss', loss, num_iters)
            t_writer.add_scalar('masked_token_loss', masked_token_loss, num_iters)
            t_writer.add_scalar('masked_response_loss', masked_res_loss, num_iters)
            t_writer.add_scalar('ent_matching_loss', ent_matching_loss, num_iters)
            t_writer.add_scalar('masked_score', batch_masked_score, num_iters)
            t_writer.add_scalar('ent_score', batch_ent_score, num_iters)
            t_writer.add_scalar('masked_res_score', batch_masked_res_score, num_iters)
            t_writer.add_scalar('lr', lr, num_iters)

            num_iters += 1

            if num_iters % args.test_iter == 0:
                eval_step += 1
                generate = eval_step%50==0
                predict_file = os.path.join(save_folder, f"eval_out_{num_iters}.tsv")
                curr_val_loss, _, _, _ = test(args, val_loader, val_gen_loader, model, tokenizer, predict_file,
                                         logging, max_val_batches, generate, device)
                if curr_val_loss <= best_loss:
                    best_loss = curr_val_loss
                    torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val.pt"))
                    et = 0
                else:
                    et += 1

                model.train()
        e += 1
        logging.info(f"Finish Training for {e} Epochs..")

        if e % args.log_epoch == 0:
            masked_ppl = math.exp(min(np.mean(masked_token_losses), MIN_PPL_THR))
            logging.info("========================= Training Status =========================")
            logging.info("Epoch                     : {}".format(e))
            logging.info("Avg. Loss                 : {:.4f}".format(np.mean(losses)))
            logging.info("Avg. Masked Token Loss    : {:.4f}".format(np.mean(masked_token_losses)))
            logging.info("Avg. Masked PPL           : {:.4f}".format(np.mean(masked_ppl)))
            logging.info("Avg. Masked Response Loss : {:.4f}".format(np.mean(masked_res_losses)))
            logging.info("Avg. Ent Matching Loss    : {:.4f}".format(np.mean(ent_matching_losses)))
            logging.info("Avg. Masked Acc.          : {:.4f}".format(np.mean(batch_masked_scores)))
            logging.info("Avg. Ent Acc.             : {:.4f}".format(np.mean(batch_ent_scores)))
            logging.info("Avg. Masked Response Acc. : {:.4f}".format(np.mean(batch_masked_res_scores)))

        if et >= args.early_stop:
            logging.info("Early Stopping..")
            break

        if (args.num_train_epochs is not None) and (e >= args.num_train_epochs):
            break

    ## evaluate at the end of training
    # predict_file = os.path.join(save_folder, f"eval_out_{num_iters}.tsv")
    # _ = test(args, val_loader, model, tokenizer, predict_file,
    #          logging, max_val_batches, device)

def test_UniLM(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
    args.test_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    experiment = "test"
    # save_folder = os.path.join(args.eval_model_dir, experiment)
    save_folder = os.path.join(args.output_dir, args.evaluate_cache)
    # os.makedirs(save_folder, exist_ok=True)
    logging_file = f"testing_{now.month}.{now.day}.log"
    predict_file = os.path.join(save_folder, f"test_out-best_eval-topk{args.top_k}-topp{args.top_p}.tsv")
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    ## prepare model & resize tokenizer
    config = BertConfig.from_pretrained(args.unilm_cache)
    config = model_utils.sort_config(config, args)
    tokenizer = BertTokenizer.from_pretrained(args.unilm_cache)
    model = BertForMMDialog(config, args.img_add_pos)

    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    bert_embedding_weight = model.bert.embeddings.word_embeddings.weight
    model.vocab_size = bert_embedding_weight.size(0)

    state = torch.load(os.path.join(args.output_dir, args.evaluate_cache, "model_best_val.pt"))
    model.load_state_dict(state)
    print("Finish Loading Model Weight")
    del state
    model.to(device)

    p_test_data = model_utils.jsonload(args.testloader_file)[0]
    if args.dataset == "WOW":
        test_episodes = json.load(open(args.episodes_test_file))
    elif args.dataset == "DD":
        test_episodes = model_utils.jsonload(args.episodes_test_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
    else:
        ent_clip_fea_file = None
    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])

    random_token_list = model_utils.gen_random_word_list(tokenizer, config.vocab_size)
    test_data = MMDialogDataset(test_episodes, device,
                               p_test_data["entity_turn"],
                               p_test_data["caption_turn"],
                               p_test_data["entities_img_feats_ids"][0],
                               p_test_data["turn_img_feats_ids"],
                               turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                               img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                               history_in_context=args.history_in_context,
                               max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                               max_turn_img_seq_length=args.max_turn_img_length,
                               per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                               max_ent_img_seq_length=args.max_ent_img_seq_length,
                               max_seq_length=args.max_seq_length,
                               max_seq_a_length=args.max_seq_a_length, is_train=True,
                               response_sep="[SEP_0]", context_sep="[SEP_1]",  caption_sep="[SEP_2]",
                               random_token_list=random_token_list,
                               test=True, dataset=args.dataset, add_ent=args.add_textual_ent,
                               add_cap=args.add_textual_cap,
                               no_ent_vis=args.no_ent_vis,
                               no_turn_vis=args.no_turn_vis)
    test_gen_data = MMDialogDataset(test_episodes, device,
                                p_test_data["entity_turn"],
                                p_test_data["caption_turn"],
                                p_test_data["entities_img_feats_ids"][0],
                                p_test_data["turn_img_feats_ids"],
                                turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                history_in_context=args.history_in_context,
                                max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                max_turn_img_seq_length=args.max_turn_img_length,
                                per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                max_ent_img_seq_length=args.max_ent_img_seq_length,
                                max_seq_length=args.max_seq_length,
                                max_seq_a_length=args.max_seq_a_length, is_train=False,
                                response_sep="[SEP_0]", context_sep="[SEP_1]",  caption_sep="[SEP_2]",
                                test=True, dataset=args.dataset, add_ent=args.add_textual_ent,
                                add_cap=args.add_textual_cap,
                                no_ent_vis=args.no_ent_vis,
                                no_turn_vis=args.no_turn_vis)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data,
                            sampler=test_sampler,
                            batch_size=args.test_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)
    gen_loader = DataLoader(test_gen_data,
                             sampler=test_sampler,
                             batch_size=args.test_batch_size,
                             pin_memory=False,
                             drop_last=False,
                             num_workers=args.workers)

    logging.info("Begin testing..")
    test(args, test_loader, gen_loader, model, tokenizer, predict_file, logging, args.max_val_batches, True, device)

def train_T5(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    else:
        args.train_batch_size = args.per_gpu_train_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    loading = True if args.load is not None else False

    t5_version = args.t5_model_name
    ex_t5 = t5_version.replace("/", "-")

    if args.no_ent_vis and args.no_turn_vis:
        ex_appended = f"novis_{now.month}.{now.day}"
    elif args.no_ent_vis:
        ex_appended = f"no_evis_{now.month}.{now.day}"
    elif args.no_turn_vis:
        ex_appended = f"no_tvis_{now.month}.{now.day}"
    else:
        ex_appended = f"{now.month}.{now.day}"
    if args.last_res:
        ex_appended = f"lastres_{ex_appended}"
    eda_append = int(args.eda) if not args.eda else f"-{args.eda_pos}"
    experiment = f"train_{ex_t5}_{args.dataset}_{args.img_add_pos}_e{args.num_train_epochs}" \
                 f"_res-mlm-{args.t5_mlm}_Ebias-{args.add_ent_bias}_lr{args.learning_rate}" \
                 f"_seqlen{args.max_seq_length}_res{args.max_seq_a_length}_bs{args.per_gpu_train_batch_size}" \
                 f"_load{loading}_cap{int(args.add_textual_cap)}_ent{int(args.add_textual_ent)}_" \
                 f"eda{eda_append}_kl{args.knowledge_len}_ev{args.ent_img_num}_{ex_appended}"

    save_folder = os.path.join(args.output_dir, experiment)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    logging_file = f"train_t5_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    ## prepare model & resize tokenizer
    config = T5Config.from_pretrained(t5_version, cache_dir=cache_dir)
    config = model_utils.sort_config(config, args)
    tokenizer = T5Tokenizer.from_pretrained(t5_version, cache_dir=cache_dir)

    special_tokens_dict = {'mask_token': '<mask>', 'sep_token': '<sep>', 'bos_token': '<s>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f'We have added {num_added_toks} tokens to the model..')

    model = T5ForMMDialogGeneration.from_pretrained(t5_version, cache_dir=cache_dir, config=config)
    model._resize_token_embeddings(len(tokenizer))

    if args.load is not None:
        print("Loading Trained Model Weight...")
        state = torch.load(os.path.join("output", args.load, "model_best_val.pt"))
        model.load_state_dict(state)
        del state

    model.to(device)
    model.train()

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # if len(args.gpu) > 1:
    #     model = nn.DataParallel(model)

    p_train_data = model_utils.jsonload(args.trainloader_file)[0]
    p_val_data = model_utils.jsonload(args.valloader_file)[0]
    if args.dataset == "WOW":
        dialog_episodes = json.load(open(args.episodes_train_file))
        val_episodes = json.load(open(args.episodes_val_file))
    elif args.dataset == "DD":
        dialog_episodes = model_utils.jsonload(args.episodes_train_file)
        val_episodes = model_utils.jsonload(args.episodes_val_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
        num_aug = 9
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
        num_aug = 5
    else:
        ent_clip_fea_file = None
        num_aug = 0

    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])
    ent_img_feats = [ent_img_feats]
    if args.ent_img_num > 1:
        for count in range(1, args.ent_img_num):
            ## lets only consider DD for now
            ent_img_feats.append(model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data/dd/processed_img_features",
                                                                        f"img_clip_features-{count+1}.pt")]))

    random_token_list = model_utils.gen_random_word_list_t5(tokenizer, len(tokenizer))

    train_data = MMDialogDataset(dialog_episodes, device,
                                 p_train_data["entity_turn"],
                                 p_train_data["caption_turn"],
                                 p_train_data["entities_img_feats_ids"][0],
                                 p_train_data["turn_img_feats_ids"],
                                 turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                 img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                 history_in_context=args.history_in_context,
                                 max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                 max_turn_img_seq_length=args.max_turn_img_length,
                                 per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                 max_ent_img_seq_length=args.max_ent_img_seq_length,
                                 max_seq_length=args.max_seq_length,
                                 max_seq_a_length=args.max_seq_a_length, is_train=True,
                                 mask_response_prob=args.mask_response_prob, mask_prob=0,
                                 random_token_list=random_token_list, num_aug=num_aug,
                                 max_masked_response_tokens=args.max_masked_response_tokens,
                                 max_masked_context_tokens=0, test=args.last_res,
                                 dataset=args.dataset, add_cap=args.add_textual_cap,
                                 add_ent=args.add_textual_ent, seq2seq=True, EDA=args.eda,
                                 eda_pos=args.eda_pos, no_ent_vis=args.no_ent_vis,
                                 no_turn_vis=args.no_turn_vis, knowledge_len=args.knowledge_len)
    val_data = MMDialogDataset(val_episodes, device,
                                p_val_data["entity_turn"],
                                p_val_data["caption_turn"],
                                p_val_data["entities_img_feats_ids"][0],
                                p_val_data["turn_img_feats_ids"],
                                turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                history_in_context=args.history_in_context,
                                max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                max_turn_img_seq_length=args.max_turn_img_length,
                                per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                max_ent_img_seq_length=args.max_ent_img_seq_length,
                                max_seq_length=args.max_seq_length,
                                max_seq_a_length=args.max_seq_a_length, is_train=False,
                                test=True, dataset=args.dataset, EDA=False,
                                add_cap=args.add_textual_cap,
                                add_ent=args.add_textual_ent,
                                seq2seq=True, no_ent_vis=args.no_ent_vis,
                                no_turn_vis=args.no_turn_vis,
                                knowledge_len=args.knowledge_len)

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              pin_memory=False,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=True)

    val_sampler = SequentialSampler(val_data)
    val_loader = DataLoader(val_data,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=True, weight_decay=args.weight_decay)
    if args.num_train_epochs is None:
        total_step = args.iterations
    else:
        total_step = len(train_loader) * args.num_train_epochs
        logging.info(f"Ignoring assigned iterations, total step is {total_step}..")
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    else:
        raise NotImplementedError

    logging.info("Begin training steps")
    logging.info("Total step: %d" % total_step)
    e = 0  # number of epoch
    num_iters = 0
    best_loss = 9e9
    best_bleu1 = 0.
    et = 0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    while (num_iters < total_step) and (et < args.early_stop):
        # Run epoch
        st = time.time()
        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n===========================================================================')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        ## record for every epoch losses
        losses, mlm_losses, lm_losses = [], [], []
        for i, data_dict in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            attention_mask = data_dict["attention_mask"][0]
            img_attention_mask = data_dict["attention_mask"][1] if args.img_add_pos != "concat" else None

            ## remove padding tokens
            masked_ids = data_dict["masked_ids"]
            masked_ids = masked_ids[masked_ids != tokenizer.pad_token_id]

            decoder_input_ids = data_dict["decoder_input_ids"][:, :-1]
            labels = data_dict["decoder_input_ids"][:, 1:].clone()
            labels = torch.where(data_dict["decoder_input_ids"][:, 1:]==tokenizer.pad_token_id, -100, labels)
            add_text_bias_flag = args.add_ent_bias or args.add_cap_bias
            seq_length = data_dict["input_ids"].shape[-1]

            input_ids = data_dict["input_ids"] if not args.eda else data_dict["input_ids_eda"]

            inputs = {"input_ids": input_ids,
                      "turn_img_feats": data_dict["turn_img_feats"] if not args.no_turn_vis else None,
                      "ent_img_feats": data_dict["ent_img_feats"]if not args.no_ent_vis else None,
                      "attention_mask": attention_mask,
                      "img_attention_mask": img_attention_mask,
                      "decoder_input_ids": decoder_input_ids,
                      "masked_decoder_input_ids": data_dict["masked_decoder_input_ids"][:, :-1],
                      "decoder_attention_mask": None,
                      "labels": labels,
                      "masked_ids": masked_ids,
                      "masked_res_pos": data_dict["masked_res_pos"],
                      "add_text_bias": add_text_bias_flag,
                      "mlm_regression": args.t5_mlm,
                      "cap_pos": data_dict["cap_pos"] if not args.no_turn_vis else None,
                      "ent_pos": data_dict["ent_pos"] if not args.no_ent_vis else None,
                      "seq_length": seq_length,
                      "return_dict": False
                      }
            with torch.cuda.amp.autocast():
                model_output = model(**inputs)
                lm_loss, mlm_loss = model_output[:2]
                loss = args.alpha * lm_loss + (1 - args.alpha) * mlm_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if args.gpu and args.n_gpu > 1:
                loss = loss.mean()
                lm_loss = lm_loss.mean()
                mlm_loss = mlm_loss.mean()

            losses.append(loss.item())
            lm_losses.append(lm_loss.item())
            mlm_losses.append(mlm_loss.item())
            lr = scheduler.get_last_lr()[0]

            # Log to Tensorboard
            t_writer.add_scalar('loss', loss.item(), num_iters)
            t_writer.add_scalar('lm_loss', lm_loss.item(), num_iters)
            t_writer.add_scalar('mlm_loss', mlm_loss.item(), num_iters)
            t_writer.add_scalar('lr', lr, num_iters)

            num_iters += 1

            if num_iters % args.test_iter == 0:
                predict_file = os.path.join(save_folder, f"eval_out_{num_iters}.tsv")
                # _ = test(args, val_loader, model, tokenizer, predict_file,
                #                 logging, max_val_batches, device)
                curr_val_loss, curr_bleu1 = evaluate_T5(args, val_loader, model, tokenizer, predict_file,
                                                        logging, v_writer, num_iters, device)

                if args.bleu_eval:
                    if curr_bleu1 > best_bleu1:
                        best_bleu1 = curr_bleu1
                        torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val_bleu.pt"))
                        et = 0
                    else:
                        et += 1
                else:
                    if curr_val_loss < best_loss:
                        best_loss = curr_val_loss
                        torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val.pt"))
                    else:
                        et += 1
                model.train()
        e += 1
        logging.info(f"Finish Training for {e} Epochs..")

        if e % args.log_epoch == 0:
            logging.info("========================= Training Status =========================")
            logging.info("Epoch               : {}".format(e))
            logging.info("Avg. Loss           : {:.4f}".format(np.mean(losses)))
            logging.info("Avg. LM Loss        : {:.4f}".format(np.mean(lm_losses)))
            logging.info("Avg. Masked LM Loss : {:.4f}".format(np.mean(mlm_losses)))

        if et >= args.early_stop:
            logging.info("Early Stopping..")
            break

        if (args.num_train_epochs is not None) and (e >= args.num_train_epochs):
            break

def evaluate_T5(args, test_dataloader, model, tokenizer, predict_file, logging, v_writer, num_iters, device):
    model.eval()
    test_step = min(len(test_dataloader), args.max_test_batches)
    cal_ppl_with_ce = lambda x: math.exp(min(x, MIN_PPL_THR))

    with torch.no_grad():
        val_losses, val_mlm_losses, val_lm_losses= [], [], []
        for step, batch in tqdm(enumerate(test_dataloader),
                                ncols=100, desc="Testing Forward: ",
                                total=test_step):

            # batch = tuple(t.to(device) for t in batch)
            img_attention_mask = batch["attention_mask"][1] if args.img_add_pos != "concat" else None

            decoder_input_ids = batch["decoder_input_ids"][:, :-1]
            labels = batch["decoder_input_ids"][:, 1:].clone()
            labels = torch.where(batch["decoder_input_ids"][:, 1:] == tokenizer.pad_token_id, -100, labels)

            inputs = {'input_ids': batch['input_ids'],
                      'decoder_input_ids': decoder_input_ids,
                      'attention_mask': batch['attention_mask'][0],
                      'turn_img_feats': batch['turn_img_feats'] if not args.no_turn_vis else None,
                      'ent_img_feats': batch['ent_img_feats'] if not args.no_ent_vis else None,
                      'img_attention_mask': img_attention_mask,
                      "cap_pos": batch["cap_pos"],
                      "ent_pos": batch["ent_pos"],
                      "labels": labels,
                      "return_dict": False
                      }

            model_output = model(**inputs)

            lm_loss, mlm_loss = model_output[:2]
            loss = args.alpha * lm_loss + (1 - args.alpha) * mlm_loss
            val_losses.append(loss.item())
            val_lm_losses.append(lm_loss.item())
            val_ppl = cal_ppl_with_ce(lm_loss.item())
            if v_writer is not None:
                # Log to Tensorboard
                v_writer.add_scalar('loss', loss, num_iters)
                v_writer.add_scalar('lm_loss', lm_loss, num_iters)
                v_writer.add_scalar('ppl', val_ppl, num_iters)

        val_ce_loss = np.mean(val_lm_losses)
        val_ppl = cal_ppl_with_ce(val_ce_loss)
        curr_val_loss = np.mean(val_losses)
        logging.info("========================= Validate Status =========================")
        logging.info("Avg. Val Loss           : {:.4f}".format(curr_val_loss))
        logging.info("Avg. Val LM Loss        : {:.4f}".format(val_ce_loss))
        logging.info("Avg. Val PPL            : {:.4f}".format(val_ppl))

    gen_word_lens = []
    gen_bpe_lens = []
    def gen_rows():
        time_meter = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader),
                                    ncols=100, desc="Testing Forward: ",
                                    total=test_step):

                # batch = tuple(t.to(device) for t in batch)
                img_attention_mask = batch["attention_mask"][1] if args.img_add_pos != "concat" else None

                encoder_inputs = {"input_ids": batch["input_ids"],
                                  "attention_mask": batch["attention_mask"][0],
                                  "turn_img_feats": batch["turn_img_feats"],
                                  "ent_img_feats": batch["ent_img_feats"],
                                  "img_attention_mask": img_attention_mask}
                encoder_outputs = model.encoder(**encoder_inputs)

                inputs = {"input_ids": batch["input_ids"],
                          "attention_mask": batch["attention_mask"][0],
                          "turn_img_feats": batch["turn_img_feats"] if not args.no_turn_vis else None,
                          "ent_img_feats": batch["ent_img_feats"] if not args.no_ent_vis else None,
                          "img_attention_mask": img_attention_mask,
                          # for adding entity labels
                          "cap_pos": batch["cap_pos"],
                          "ent_pos": batch["ent_pos"],
                          "seq_length": batch["input_ids"].shape[-1],

                          "encoder_outputs": encoder_outputs,

                          'do_sample': args.do_sample,
                          # hyperparameters of beam search
                          "max_length": args.max_seq_a_length,
                          "num_beams": args.num_beams if args.img_add_pos == "concat" else 1, ## beams can only be 1 for now, otherwise need
                          "temperature": args.temperature,
                          "top_k": args.top_k,
                          "top_p": args.top_p,
                          "repetition_penalty": args.repetition_penalty,
                          "length_penalty": args.length_penalty,
                          "num_return_sequences": args.num_return_sequences,
                          "num_keep_best": args.num_keep_best,
                          }
                tic = time.time()
                # captions, logprobs
                output_sequences = model.generate(**inputs)

                all_responses = batch["response"]
                ctext = [s[s.find("</s>") + len("</s>"):] for s in all_responses]
                words_len = sum([len(
                    [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                    s in ctext])
                gen_word_lens.append(words_len)
                ctext_ids = tokenizer(all_responses)['input_ids']
                tokens = [t[:t.index(tokenizer.eos_token_id) + 1] if tokenizer.eos_token_id in t else t for t in ctext_ids]
                words_bpe = sum([len(t) for t in tokens])
                gen_bpe_lens.append(words_bpe)

                all_caps = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                time_meter += time.time() - tic

                for caps, response in zip(all_caps, all_responses):
                    res = []
                    # for cap in caps:
                    # cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                    # cap = cap.split('.')[0]
                    res.append({'response_pred': caps, 'response': response})
                    yield res
                    # return res
                if step >= test_step:
                    break
        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    model_utils.tsv_writer(gen_rows(), predict_file)

    bleu1 = evaluate_on_wow_multimodal_training(predict_file)['Bleu_1']

    assert len(val_lm_losses)==len(gen_word_lens)==len(gen_bpe_lens)
    val_loader_len = len(val_lm_losses)
    val_ppl_word = cal_ppl_with_ce(np.sum([val_lm_losses[i] * gen_word_lens[i] for i in range(val_loader_len)]) / np.sum(gen_word_lens))
    val_ppl_bpe = cal_ppl_with_ce(np.sum([val_lm_losses[i] * gen_bpe_lens[i] for i in range(val_loader_len)]) / np.sum(gen_bpe_lens))
    logging.info("Val Word PPL            : {:.4f}".format(val_ppl_word))
    logging.info("Val BPE PPL             : {:.4f}".format(val_ppl_bpe))
    logging.info("Val BLEU 1              : {:.4f}".format(bleu1))

    return curr_val_loss, bleu1

def test_T5(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
    args.test_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # save_folder = os.path.join(args.eval_model_dir, experiment)
    save_folder = os.path.join(args.output_dir, args.evaluate_cache)
    # os.makedirs(save_folder, exist_ok=True)
    logging_file = f"testing_{now.month}.{now.day}.log"
    predict_file = os.path.join(save_folder, f"test_out-best_eval-topk{args.top_k}-topp{args.top_p}.tsv")
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    t5_version = args.t5_model_name

    ## prepare model & resize tokenizer
    config = T5Config.from_pretrained(t5_version, cache_dir=cache_dir)
    config = model_utils.sort_config(config, args)
    tokenizer = T5Tokenizer.from_pretrained(t5_version, cache_dir=cache_dir)

    special_tokens_dict = {'mask_token': '<mask>', 'sep_token': '<sep>', 'bos_token': '<s>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f'We have added {num_added_toks} tokens to the model..')

    model = T5ForMMDialogGeneration.from_pretrained(t5_version, cache_dir=cache_dir, config=config)
    model._resize_token_embeddings(len(tokenizer))

    state = torch.load(os.path.join(args.output_dir, args.evaluate_cache, "model_best_val.pt"))
    model.load_state_dict(state, strict=False)
    print("Finish Loading Trained Model Weight..")
    del state
    model.to(device)

    p_test_data = model_utils.jsonload(args.testloader_file)[0]
    if args.dataset == "WOW":
        test_episodes = json.load(open(args.episodes_test_file))
    elif args.dataset == "DD":
        test_episodes = model_utils.jsonload(args.episodes_test_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
    else:
        ent_clip_fea_file = None
    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])

    ent_img_feats = [ent_img_feats]
    if args.ent_img_num > 1:
        for count in range(1, args.ent_img_num):
            ## lets only consider DD for now
            ent_img_feats.append(model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data/dd/processed_img_features",
                                                                        f"img_clip_features-{count + 1}.pt")]))

    test_data = MMDialogDataset(test_episodes, device,
                               p_test_data["entity_turn"],
                               p_test_data["caption_turn"],
                               p_test_data["entities_img_feats_ids"][0],
                               p_test_data["turn_img_feats_ids"],
                               turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                               img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                               history_in_context=args.history_in_context,
                               max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                               max_turn_img_seq_length=args.max_turn_img_length,
                               per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                               max_ent_img_seq_length=args.max_ent_img_seq_length,
                               max_seq_length=args.max_seq_length, EDA=False,
                               max_seq_a_length=args.max_seq_a_length, is_train=False,
                               test=True, dataset=args.dataset, add_ent=args.add_textual_ent,
                               add_cap=args.add_textual_cap, seq2seq=True,
                               no_ent_vis=args.no_ent_vis, no_turn_vis=args.no_turn_vis)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data,
                            sampler=test_sampler,
                            batch_size=args.test_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)

    logging.info("Begin testing..")
    _, _ = evaluate_T5(args, test_loader, model, tokenizer, predict_file, logging, None, 0, device)


def train_gpt(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    else:
        args.train_batch_size = args.per_gpu_train_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    loading = True if args.load is not None else False

    gpt_version = args.gpt_model_name
    ex_gpt = gpt_version.replace("/", "-")

    if args.no_ent_vis and args.no_turn_vis:
        ex_appended = f"novis_{now.month}.{now.day}"
    elif args.no_ent_vis:
        ex_appended = f"no_evis_{now.month}.{now.day}"
    elif args.no_turn_vis:
        ex_appended = f"no_tvis_{now.month}.{now.day}"
    else:
        ex_appended = f"{now.month}.{now.day}"
    if args.last_res:
        ex_appended = f"lastres_{ex_appended}"
    experiment = f"train_{ex_gpt}_{args.dataset}_{args.img_add_pos}_e{args.num_train_epochs}" \
                 f"_res-mlm-{args.t5_mlm}_Ebias-{args.add_ent_bias}_lr{args.learning_rate}" \
                 f"_seqlen{args.max_seq_length}_bs{args.per_gpu_train_batch_size}_load{loading}_" \
                 f"cap{int(args.add_textual_cap)}_ent{int(args.add_textual_ent)}_eda{int(args.eda)}" \
                 f"_kl{args.knowledge_len}_{ex_appended}"

    save_folder = os.path.join(args.output_dir, experiment)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    logging_file = f"train_gpt_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    ## prepare model & resize tokenizer
    config = GPT2Config.from_pretrained(gpt_version, cache_dir=cache_dir)
    config = model_utils.sort_config(config, args)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_version, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    # special_tokens_dict = {'mask_token': '<mask>', 'sep_token': '<sep>'}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # logging.info(f'We have added {num_added_toks} tokens to the model..')

    model = GPT2LMforMMDialog.from_pretrained(gpt_version, cache_dir=cache_dir, config=config)
    # model.resize_token_embeddings(len(tokenizer)

    if args.load is not None:
        print("Loading Trained Model Weight...")
        state = torch.load(os.path.join("output", args.load, "model_best_val.pt"))
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        model.load_state_dict(state)
        del state

    if args.dp:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.train()

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # if len(args.gpu) > 1:
    #     model = nn.DataParallel(model)

    p_train_data = model_utils.jsonload(args.trainloader_file)[0]
    p_val_data = model_utils.jsonload(args.valloader_file)[0]
    if args.dataset == "WOW":
        dialog_episodes = json.load(open(args.episodes_train_file))
        val_episodes = json.load(open(args.episodes_val_file))
    elif args.dataset == "DD":
        dialog_episodes = model_utils.jsonload(args.episodes_train_file)
        val_episodes = model_utils.jsonload(args.episodes_val_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
        num_aug = 9
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
        num_aug = 5
    else:
        ent_clip_fea_file = None
        num_aug = 0

    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])
    random_token_list = None

    train_data = MMDialogDataset(dialog_episodes, device,
                                 p_train_data["entity_turn"],
                                 p_train_data["caption_turn"],
                                 p_train_data["entities_img_feats_ids"][0],
                                 p_train_data["turn_img_feats_ids"],
                                 turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                 img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                 history_in_context=args.history_in_context,
                                 max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                 max_turn_img_seq_length=args.max_turn_img_length,
                                 per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                 max_ent_img_seq_length=args.max_ent_img_seq_length,
                                 max_seq_length=args.max_seq_length,
                                 max_seq_a_length=args.max_seq_a_length, is_train=True,
                                 mask_response_prob=args.mask_response_prob, mask_prob=0,
                                 random_token_list=random_token_list, num_aug=num_aug,
                                 max_masked_response_tokens=args.max_masked_response_tokens,
                                 max_masked_context_tokens=0, test=False,
                                 dataset=args.dataset, add_cap=args.add_textual_cap,
                                 add_ent=args.add_textual_ent, use_gpt=True, EDA=args.eda,
                                 no_ent_vis=args.no_ent_vis, no_turn_vis=args.no_turn_vis,
                                 knowledge_len=args.knowledge_len)
    val_data = MMDialogDataset(val_episodes, device,
                                p_val_data["entity_turn"],
                                p_val_data["caption_turn"],
                                p_val_data["entities_img_feats_ids"][0],
                                p_val_data["turn_img_feats_ids"],
                                turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                history_in_context=args.history_in_context,
                                max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                max_turn_img_seq_length=args.max_turn_img_length,
                                per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                max_ent_img_seq_length=args.max_ent_img_seq_length,
                                max_seq_length=args.max_seq_length,
                                max_seq_a_length=args.max_seq_a_length, is_train=False,
                                test=True, dataset=args.dataset,
                                add_cap=args.add_textual_cap,
                                add_ent=args.add_textual_ent, EDA=False,
                                use_gpt=True, no_ent_vis=args.no_ent_vis,
                                no_turn_vis=args.no_turn_vis,
                                knowledge_len=args.knowledge_len)

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              pin_memory=False,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=True)

    val_sampler = SequentialSampler(val_data)
    val_loader = DataLoader(val_data,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=args.workers)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.num_train_epochs is None:
        total_step = args.iterations
    else:
        total_step = len(train_loader) * args.num_train_epochs
        logging.info(f"Ignoring assigned iterations, total step is {total_step}..")
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_step,
                                                    num_training_steps=total_step)
    else:
        raise NotImplementedError

    logging.info("Begin training steps")
    logging.info("Total step: %d" % total_step)
    e = 0  # number of epoch
    num_iters = 0
    best_loss = 9e9
    best_bleu1 = 0.
    et = 0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    while (num_iters < total_step) and (et < args.early_stop):
        # Run epoch
        st = time.time()
        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n===========================================================================')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        ## record for every epoch losses
        losses = []
        for i, data_dict in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            add_text_bias_flag = args.add_ent_bias or args.add_cap_bias

            input_ids = data_dict["input_ids"] if not args.eda else data_dict["input_ids_eda"]

            inputs = {"input_ids": input_ids,
                      "turn_img_feats": data_dict["turn_img_feats"] if not args.no_turn_vis else None,
                      "ent_img_feats": data_dict["ent_img_feats"] if not args.no_ent_vis else None,
                      "attention_mask": data_dict["attention_mask"][0],
                      "labels": data_dict["input_ids"],
                      "add_text_bias": add_text_bias_flag,
                      "ent_pos": data_dict["ent_pos"] if not args.no_ent_vis else None,
                      "token_type_ids": data_dict["segment_ids"],
                      "return_dict": False
                      }
            with torch.cuda.amp.autocast():
                model_output = model(**inputs)
                loss = model_output[0]
                if args.dp:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            lr = scheduler.get_last_lr()[0]

            # Log to Tensorboard
            t_writer.add_scalar('loss', loss.item(), num_iters)
            t_writer.add_scalar('lr', lr, num_iters)

            num_iters += 1

            if num_iters % args.test_iter == 0:
                predict_file = os.path.join(save_folder, f"eval_out_{num_iters}.tsv")
                curr_val_loss, curr_bleu1 = evaluate_gpt(args, val_loader, model, tokenizer, predict_file,
                                                        logging, v_writer, num_iters, device, gen=False)

                if args.bleu_eval:
                    if curr_bleu1 > best_bleu1:
                        best_bleu1 = curr_bleu1
                        torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val_bleu.pt"))
                        et = 0
                    else:
                        et += 1
                else:
                    if curr_val_loss < best_loss:
                        best_loss = curr_val_loss
                        torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val.pt"))
                    else:
                        et += 1
                model.train()
        e += 1
        logging.info(f"Finish Training for {e} Epochs..")

        if e % args.log_epoch == 0:
            logging.info("========================= Training Status =========================")
            logging.info("Epoch               : {}".format(e))
            logging.info("Avg. Loss           : {:.4f}".format(np.mean(losses)))

        if et >= args.early_stop:
            logging.info("Early Stopping..")
            break

        if (args.num_train_epochs is not None) and (e >= args.num_train_epochs):
            break

def evaluate_gpt(args, test_dataloader, model, tokenizer, predict_file, logging, v_writer, num_iters, device, gen=True):
    model.eval()
    test_step = min(len(test_dataloader), args.max_test_batches)
    cal_ppl_with_ce = lambda x: math.exp(min(x, MIN_PPL_THR))

    with torch.no_grad():
        val_losses = []
        for step, batch in tqdm(enumerate(test_dataloader),
                                ncols=100, desc="Testing Forward: ",
                                total=test_step):

            # batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'][0],
                      'turn_img_feats': batch['turn_img_feats'] if not args.no_turn_vis else None,
                      'ent_img_feats': batch['ent_img_feats'] if not args.no_ent_vis else None,
                      "ent_pos": batch["ent_pos"],
                      "labels": batch['input_ids'],
                      "token_type_ids": batch["segment_ids"],
                      "return_dict": False
                      }

            model_output = model(**inputs)

            loss = model_output[0]
            if args.dp:
                loss = loss.mean()
            val_losses.append(loss.item())
            val_ppl = cal_ppl_with_ce(loss.item())
            if v_writer is not None:
                # Log to Tensorboard
                v_writer.add_scalar('loss', loss, num_iters)
                v_writer.add_scalar('ppl', val_ppl, num_iters)

        val_ce_loss = np.mean(val_losses)
        val_ppl = cal_ppl_with_ce(val_ce_loss)
        curr_val_loss = np.mean(val_losses)
        logging.info("========================= Validate Status =========================")
        logging.info("Avg. Val Loss           : {:.4f}".format(curr_val_loss))
        logging.info("Avg. Val PPL            : {:.4f}".format(val_ppl))

    def gen_rows():
        time_meter = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader),
                                    ncols=100, desc="Testing Forward: ",
                                    total=test_step):

                inputs = {"input_ids": batch["input_ids"],
                          "attention_mask": batch["attention_mask"][0],
                          "turn_img_feats": batch["turn_img_feats"] if not args.no_turn_vis else None,
                          "ent_img_feats": batch["ent_img_feats"] if not args.no_ent_vis else None,
                          # for adding entity labels
                          "ent_pos": batch["ent_pos"],
                          "token_type_ids": batch["segment_ids"],

                          'do_sample': args.do_sample,
                          # hyperparameters of beam search
                          "max_length": FIXED_GPT_LEN,
                          # "num_beams": args.num_beams,
                          "temperature": args.temperature,
                          "top_k": args.top_k,
                          "top_p": args.top_p,
                          "repetition_penalty": args.repetition_penalty,
                          "length_penalty": args.length_penalty,
                          "num_return_sequences": args.num_return_sequences,
                          "num_keep_best": args.num_keep_best,
                          }
                tic = time.time()

                if args.dp:
                    output_sequences = model.module.generate(**inputs)
                else:
                    output_sequences = model.generate(**inputs)

                time_meter += time.time() - tic

                for i, response_pred in enumerate(output_sequences):
                    res = []
                    response_pred = tokenizer.decode(response_pred[batch['input_ids'][i].shape[-1] + 1:
                                                                   batch['input_ids'][i].shape[-1] + args.max_seq_a_length + 1],
                                                     skip_special_tokens=True)
                    res.append({'response_pred': response_pred, 'response': batch['response'][i]})
                    yield res
                    # return res
                if step >= test_step:
                    break
        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    if gen:
        model_utils.tsv_writer(gen_rows(), predict_file)
        bleu1 = evaluate_on_wow_multimodal_training(predict_file)['Bleu_1']

    bleu1=0.
    val_ppl = cal_ppl_with_ce(np.mean(val_losses))
    logging.info("Val PPL            : {:.4f}".format(val_ppl))
    if gen:
        logging.info("Val BLEU 1         : {:.4f}".format(bleu1))

    return curr_val_loss, bleu1

def test_gpt(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
    args.test_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # save_folder = os.path.join(args.eval_model_dir, experiment)
    save_folder = os.path.join(args.output_dir, args.evaluate_cache)
    logging_file = f"testing_{now.month}.{now.day}.log"
    predict_file = os.path.join(save_folder, f"test_out-best_eval-topk{args.top_k}-topp{args.top_p}.tsv")
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    gpt_version = args.gpt_model_name

    ## prepare model & resize tokenizer
    config = GPT2Config.from_pretrained(gpt_version, cache_dir=cache_dir)
    config = model_utils.sort_config(config, args)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_version, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # special_tokens_dict = {'mask_token': '<mask>', 'sep_token': '<sep>'}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # logging.info(f'We have added {num_added_toks} tokens to the model..')

    model = GPT2LMforMMDialog.from_pretrained(gpt_version, cache_dir=cache_dir, config=config)
    # model._resize_token_embeddings(len(tokenizer))

    print("Loading Trained Model Weight...")
    state = torch.load(os.path.join(save_folder, "model_best_val.pt"))
    if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attribute 'module'
        state_copy = copy.copy(state)
        keys = state_copy.keys()
        for k in keys:
            state[k.replace('module.', '')] = state.pop(k)
    model.load_state_dict(state)
    del state
    model.to(device)

    p_test_data = model_utils.jsonload(args.testloader_file)[0]
    if args.dataset == "WOW":
        test_episodes = json.load(open(args.episodes_test_file))
    elif args.dataset == "DD":
        test_episodes = model_utils.jsonload(args.episodes_test_file)
    else:
        raise NotImplementedError

    fea_prefix_path = os.path.join(data_dir, f"processed_resee_data/shared/processed_img_features_clip_{args.vit_version}/")
    fea_paths = ["openimagev6_train_clip_vis_fea.pt",
                 "openimagev6_test_clip_vis_fea.pt",
                 "openimagev6_val_clip_vis_fea.pt",
                 "coco_train_clip_vis_fea.pt",
                 "coco_val_clip_vis_fea.pt",
                 "flickr30_clip_vis_fea.pt",
                 "nocaps_clip_vis_fea.pt"]
    fea_paths = [os.path.join(fea_prefix_path, item) for item in fea_paths]
    turn_img_feats = model_utils.load_pt_list(fea_paths)

    if args.dataset == "WOW":
        ent_clip_fea_file = "wow/processed_img_features/img_clip_features.pt"
    elif args.dataset == "DD":
        ent_clip_fea_file = "dd/processed_img_features/img_clip_features.pt"
    else:
        ent_clip_fea_file = None
    ent_img_feats = model_utils.load_pt_list([os.path.join(data_dir, "processed_resee_data", ent_clip_fea_file)])

    test_data = MMDialogDataset(test_episodes, device,
                                p_test_data["entity_turn"],
                                p_test_data["caption_turn"],
                                p_test_data["entities_img_feats_ids"][0],
                                p_test_data["turn_img_feats_ids"],
                                turn_img_feats=turn_img_feats, ent_img_feats=ent_img_feats,
                                img_add_pos=args.img_add_pos, img_dim=args.img_feature_dim,
                                history_in_context=args.history_in_context,
                                max_episode_length=args.max_episode_length, tokenizer=tokenizer,
                                max_turn_img_seq_length=args.max_turn_img_length,
                                per_max_turn_img_seq_length=args.per_max_turn_img_seq_length,
                                max_ent_img_seq_length=args.max_ent_img_seq_length,
                                max_seq_length=args.max_seq_length, EDA=False,
                                max_seq_a_length=args.max_seq_a_length, is_train=False,
                                test=True, dataset=args.dataset, add_ent=args.add_textual_ent,
                                add_cap=args.add_textual_cap, use_gpt=True,
                                no_ent_vis=args.no_ent_vis, no_turn_vis=args.no_turn_vis)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data,
                             sampler=test_sampler,
                             batch_size=args.test_batch_size,
                             pin_memory=False,
                             drop_last=False,
                             num_workers=args.workers)

    logging.info("Begin testing..")
    _, _ = evaluate_gpt(args, test_loader, model, tokenizer, predict_file, logging, None, 0, device, gen=True)



if __name__ == '__main__':
    args = parser.parse_args()

    ## data loading for two different tasks
    if args.dataset == "WOW":
        args.episodes_train_file = os.path.join(data_dir, "processed_resee_data/wow/train.json")
        args.episodes_val_file = os.path.join(data_dir, "processed_resee_data/wow/valid_topic_split.json")
        args.episodes_test_file = os.path.join(data_dir, "processed_resee_data/wow/test_topic_split.json")
        args.trainloader_file = os.path.join(data_dir, "processed_resee_data/wow/train_v0.json")
        args.valloader_file = os.path.join(data_dir, "processed_resee_data/wow/valid_topic_v0.json")
        args.testloader_file = os.path.join(data_dir, "processed_resee_data/wow/test_topic_v0.json")
    elif args.dataset == "DD":
        args.episodes_train_file = os.path.join(data_dir, "processed_resee_data/dd/train.json")
        args.episodes_val_file = os.path.join(data_dir, "processed_resee_data/dd/valid.json")
        args.episodes_test_file = os.path.join(data_dir, "processed_resee_data/dd/test.json")
        args.trainloader_file = os.path.join(data_dir, "processed_resee_data/dd/train_v0.json")
        args.valloader_file = os.path.join(data_dir, "processed_resee_data/dd/valid_v0.json")
        args.testloader_file = os.path.join(data_dir, "processed_resee_data/dd/test_v0.json")
    else:
        raise NotImplementedError

    if args.do_train:
        if args.model_type=="unilm":
            train_UniLM(args)
        elif args.model_type=="t5":
            train_T5(args)
        elif args.model_type=="gpt":
            train_gpt(args)
        else:
            raise NotImplementedError
    else:
        if args.model_type=="unilm":
            test_UniLM(args)
        elif args.model_type=="t5":
            test_T5(args)
        elif args.model_type=="gpt":
            test_gpt(args)
        else:
            raise NotImplementedError