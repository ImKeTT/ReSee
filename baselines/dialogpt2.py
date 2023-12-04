#!/usr/bin/env python
#-*- coding: utf-8 -*-
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import datetime, math, os, sys, json, argparse, time, re, inspect, copy
sys.path.append("../")
from tensorboardX import SummaryWriter
from logger import Logger
from utils.model_utils import jsonload, tsv_writer
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.eval_utils import evaluate_on_wow_multimodal_training

CACHE_DIR='/usr/.cache/torch/transformers'
MIN_PPL_THR=10

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='dd', type=str, required=False)

parser.add_argument("--seed", default=42, type=int, required=False)

parser.add_argument("--load", default=None, type=str, required=False)
parser.add_argument("--num_train_epochs", default=30, type=int, required=False)
parser.add_argument("--learning_rate", default=1e-3, type=float, required=False)
parser.add_argument("--weight_decay", default=1e-6, type=float, required=False)
parser.add_argument("--early_stop", default=3, type=int, required=False)
parser.add_argument("--log_epoch", default=2, type=int, required=False)
parser.add_argument("--test_iter", default=2000, type=int, required=False)
parser.add_argument("--max_length", default=90, type=int, required=False)
parser.add_argument("--response_max_len", default=30, type=int, required=False)

parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--model_name", default="microsoft/DialoGPT-medium",
                    type=str, required=False, choices=["microsoft/DialoGPT-medium", "gpt2-medium"])
parser.add_argument("--output_dir", default="outputs", type=str, required=False)


parser.add_argument('--no_gpu', action='store_true')
parser.add_argument('--truncate', action='store_true')
parser.add_argument('--no_finetune', action='store_true')
parser.add_argument('--dp', action='store_true')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--max_val_batch", default=100, type=int)
parser.add_argument("--workers", default=3, type=int,
                    help="Dataloader worker.")

class WoWDialog(Dataset):
    def __init__(self, dialog, eos_token, dataset="wow"):
        self.dialog = dialog
        self.eos_token = eos_token
        if dataset == "wow":
            self.dialog_key = 'dialog'
        elif dataset == "dd":
            self.dialog_key = 'dialogue'
        else:
            raise NotImplementedError
        self._initialization()

    def _initialization(self):
        self.dialog_history = []
        self.dialog_response = []
        for line in self.dialog:
            histroy = ''
            for i, context in enumerate(line[self.dialog_key][:-1]):
                histroy += context['text'] + self.eos_token
            self.dialog_response.append(line[self.dialog_key][-1]['text'] + self.eos_token)
            self.dialog_history.append(histroy)
        assert len(self.dialog_history)==len(self.dialog_response)

    def __getitem__(self, item):
        dialog_his = self.dialog_history[item]
        dialog_res = self.dialog_response[item]
        dialog_full = self.dialog_history[item] + self.dialog_response[item]
        return {"history": dialog_his, "response": dialog_res, "full": dialog_full}

    def __len__(self):
        return len(self.dialog)


def evaluate(args, val_loader, model, tokenizer, logging, device, save_folder, num_iters, gen=False):
    model.eval()
    losses = []
    val_step = min(len(val_loader), args.max_val_batch)
    for i, data_dict in enumerate(tqdm(val_loader, ncols=100, desc="Testing Forward: ",
                                  total=val_step)):
        tokens = data_dict["full"]
        if args.truncate:
            encoded = tokenizer(tokens, padding='max_length', truncation=True,
                                max_length=args.max_length, return_tensors="pt")
        else:
            encoded = tokenizer(tokens, padding=True, return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "labels": encoded["input_ids"].to(device)
        }
        outputs = model(**inputs)
        loss = outputs[0]
        if args.dp:
            loss = loss.mean()
        losses.append(loss.item())

    def gen_rows():
        time_meter = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(val_loader),
                                    ncols=100, desc="Testing Forward: "):

                # batch = tuple(t.to(device) for t in batch)
                tokens = batch["history"]
                if args.truncate:
                    encoded = tokenizer(tokens, padding='max_length', truncation=True,
                                                max_length=args.max_length, return_tensors="pt")
                else:
                    encoded = tokenizer(tokens, padding=True, return_tensors="pt")
                inputs = {
                    "input_ids": encoded['input_ids'].to(device),
                    "attention_mask": encoded['attention_mask'].to(device),
                    "do_sample": False,
                    "max_length": 500,
                          }
                tic = time.time()
                if args.dp:
                    responses_pred = model.module.generate(**inputs)
                else:
                    responses_pred = model.generate(**inputs)
                time_meter += time.time() - tic
                for i, response_pred in enumerate(responses_pred):
                    res = []
                    response_pred = tokenizer.decode(response_pred[encoded['input_ids'][i].shape[-1]: encoded['input_ids'][i].shape[-1] + args.response_max_len + 1], skip_special_tokens=True)
                    res.append({'response_pred': response_pred, 'response': batch["response"][i][:-len(tokenizer.eos_token)]})
                    yield res
        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    if gen:
        predict_file = os.path.join(save_folder, f"val_{num_iters}.tsv")
        tsv_writer(gen_rows(), predict_file)
        bleu1 = evaluate_on_wow_multimodal_training(predict_file)['Bleu_1']
    logging.info("========================= Validation Status =========================")
    logging.info("Avg. Val Loss   : {:.4f}".format(np.mean(losses)))
    logging.info("Avg. Val PPL    : {:.4f}".format(math.exp(min(np.mean(losses), MIN_PPL_THR))))
    if gen:
        logging.info("Avg. Val BLEU1  : {:.4f}".format(bleu1))
    logging.info("=====================================================================")
    return np.mean(losses)

def test_gpt_model(args):
    now = datetime.datetime.now()
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

    experiment = args.load
    save_folder = os.path.join(args.output_dir, experiment)
    os.makedirs(save_folder, exist_ok=True)

    if args.dataset == "wow":
        test_corpus = json.load(open(os.path.join("/data/tuhq/multimodal/wow", "test_topic_split.json")))
    elif args.dataset == "dd":
        test_corpus = jsonload(os.path.join("/data/tuhq/multimodal/dailydialog", "test.json"))
    else:
        raise NotImplementedError

    ft_flag = 0 if args.no_finetune else 1
    logging_file = f"test_{args.dataset}_ft{ft_flag}_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    prefix_pth = "/data/tuhq/multimodal/MMDialog/dialogsrc/baselines/outputs/"
    model = GPT2LMHeadModel.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)

    if not args.no_finetune:
        state = torch.load(os.path.join(prefix_pth, args.load, "model_best_val.pt"))
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        model.load_state_dict(state)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    test_data = WoWDialog(test_corpus, eos_token, args.dataset)

    test_loader = DataLoader(test_data,
                              batch_size=5,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=False)

    losses = []
    for i, data_dict in enumerate(tqdm(test_loader, ncols=100, desc="Testing Forward: ")):
        tokens = data_dict["full"]
        if args.truncate:
            encoded = tokenizer(tokens, padding='max_length', truncation=True,
                                max_length=args.max_length, return_tensors="pt")
        else:
            encoded = tokenizer(tokens, padding=True, return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "labels": encoded["input_ids"].to(device)
        }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        losses.append(loss.item())

    logging.info("========================= Testing Status =========================")
    logging.info("Avg. Test Loss   : {:.4f}".format(np.mean(losses)))
    logging.info("Avg. Test PPL    : {:.4f}".format(math.exp(min(np.mean(losses), MIN_PPL_THR))))
    logging.info("==================================================================")

    def gen_rows():
        time_meter = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_loader),
                                    ncols=100, desc="Testing Forward: "):

                # batch = tuple(t.to(device) for t in batch)
                tokens = batch["history"]
                if args.truncate:
                    encoded = tokenizer(tokens, padding='max_length', truncation=True,
                                                max_length=args.max_length, return_tensors="pt")
                else:
                    encoded = tokenizer(tokens, padding=True, return_tensors="pt")
                inputs = {
                    "input_ids": encoded['input_ids'].to(device),
                    "attention_mask": encoded['attention_mask'].to(device),
                    "do_sample": False,
                    "max_length": 500,
                          }
                tic = time.time()
                responses_pred = model.generate(**inputs)
                time_meter += time.time() - tic
                for i, response_pred in enumerate(responses_pred):
                    res = []
                    response_pred = tokenizer.decode(response_pred[encoded['input_ids'][i].shape[-1]: encoded['input_ids'][i].shape[-1] + args.response_max_len + 1], skip_special_tokens=True)
                    res.append({'response_pred': response_pred, 'response': batch["response"][i][:-len(tokenizer.eos_token)]})
                    yield res

        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    predict_file = os.path.join(save_folder, f"test_out_ft{ft_flag}_resmx{args.response_max_len}_{now.month}.{now.day}.tsv")
    tsv_writer(gen_rows(), predict_file)

def run(args):
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
    print(device)

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log_model_name = args.model_name.replace("/", "-")
    experiment = f"{log_model_name}_{args.dataset}_trunc_{args.max_length}" if args.truncate else f"{log_model_name}_{args.dataset}"
    save_folder = os.path.join(args.output_dir, experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)

    if args.dataset == "wow":
        train_corpus = json.load(open(os.path.join("/data/tuhq/multimodal/wow", "train.json")))
        val_corpus = json.load(open(os.path.join("/data/tuhq/multimodal/wow", "valid_topic_split.json")))
    elif args.dataset == "dd":
        train_corpus = jsonload(os.path.join("/data/tuhq/multimodal/dailydialog", "train.json"))
        val_corpus = jsonload(os.path.join("/data/tuhq/multimodal/dailydialog", "valid.json"))
    else:
        raise NotImplementedError

    logging_file = f"{log_model_name}_{args.dataset}_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    model = GPT2LMHeadModel.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    if args.dp:
        model = torch.nn.DataParallel(model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    model.to(device)

    train_data = WoWDialog(train_corpus, eos_token, args.dataset)
    val_data = WoWDialog(val_corpus, eos_token, args.dataset)
    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=False)
    val_loader = DataLoader(val_data,
                            batch_size=args.eval_batch_size,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=args.workers,
                            shuffle=False)

    total_step = len(train_loader) * args.num_train_epochs

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.2 * total_step,
                                                num_training_steps=total_step)

    logging.info("Begin training steps")
    logging.info("Total step: %d" % total_step)
    e = 0  # number of epoch
    num_iters = 0
    best_loss = 9e9
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
            ## remove eos token of history during training
            tokens = data_dict["full"]
            if args.truncate:
                encoded = tokenizer(tokens, padding='max_length', truncation=True,
                                    max_length=args.max_length, return_tensors="pt")
            else:
                encoded = tokenizer(tokens, padding=True, return_tensors="pt")
            inputs = {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
                "labels": encoded["input_ids"].to(device)
            }
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs[0]
                if args.dp:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

            lr = scheduler.get_last_lr()[0]
            t_writer.add_scalar('loss', loss, num_iters)
            t_writer.add_scalar('lr', lr, num_iters)

            num_iters += 1

            test_time = 0
            if num_iters % args.test_iter == 0:
                test_time += 1
                eval_loss = evaluate(args, val_loader, model, tokenizer, logging,
                                                 device, save_folder, num_iters, test_time%3==0)
                model.train()
                curr_loss = eval_loss
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val.pt"))
                    et = 0
                else:
                    et += 1

        e += 1
        logging.info(f"Finish Training for {e} Epochs..")

        if e % args.log_epoch == 0:
            logging.info("========================= Training Status =========================")
            logging.info("Epoch       : {}".format(e))
            logging.info("Avg. Loss   : {:.4f}".format(np.mean(losses)))
            logging.info("Avg. PPL    : {:.4f}".format(math.exp(min(np.mean(losses), MIN_PPL_THR))))

        if et >= args.early_stop:
            logging.info("Early Stopping.. And Saving Model Weights..")
            torch.save(model.state_dict(), os.path.join(save_folder, "model_last.pt"))
            break
        if (args.num_train_epochs is not None) and (e >= args.num_train_epochs):
            break

if __name__ == '__main__':
    args = parser.parse_args()
    # args = parser.parse_args('--dataset wow --num_train_epochs 30 --test_iter 3000 --learning_rate 5e-5 --max_length 190 --dp --truncate --per_gpu_train_batch_size 8 --model_name microsoft/DialoGPT-medium'.split())

    test_gpt_model(args)
    # run(args)


