import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Callable
from notebooks.utils import all_paths_exist
from model import SideEffectClassificationModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaConfig


DEFAULT_DATASET = 'cutoff_dataset'
DEFAULT_DATASET_DIR = f'datasets/{DEFAULT_DATASET}'
DEFAULT_TRAIN_FILE = f'{DEFAULT_DATASET_DIR}/train.csv'
DEFAULT_EVAL_FILE = f'{DEFAULT_DATASET_DIR}/eval.csv'
DEFAULT_TEST_FILE = f'{DEFAULT_DATASET_DIR}/test.csv'
DEFAULT_OUTPUT_DIR = f'{DEFAULT_DATASET}_saved_models'

DEFAULT_CONFIG_NAME = 'microsoft/graphcodebert-base'
DEFAULT_TOKENIZER = DEFAULT_CONFIG_NAME

DEFAULT_TO_TRAIN = True
DEFAULT_TO_TEST = not DEFAULT_TO_TRAIN

DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 16 # empirically found to be most time efficient
DEFAULT_EPOCHS = 10
DEFAULT_EPSILON = 1e-8
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 42

MAX_SEQ_LEN = 512
SCHEDULER_PATIENCE = 3
BEST_MODEL_WEIGHTS = 'best_model_weights.pth'

LINE_COL_IDX = 0
CODE_COL_IDX = 1
LABEL_COL_IDX = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
proj_dir = os.path.dirname(os.path.abspath('__file__'))


class CSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        line = self.df.iloc[idx, LINE_COL_IDX]
        code = self.df.iloc[idx, CODE_COL_IDX]
        label = torch.tensor(self.df.iloc[idx, LABEL_COL_IDX], device=device)
        
        return line, code, label
    
    
def tokenize_batch(batch: tuple, tokenizer: AutoTokenizer)->list[torch.Tensor]:
    token_ids = []
    attn_masks = []
    
    for i in range(len(batch[0])):
        curr_line = batch[0][i]
        curr_code = batch[1][i]
        
        # UNRESOLVED ISSUE: Our problem is highly context dependent
        # and truncating is necessary for the GraphCodeBERT model
        # if the token sequence is too long.
        line_tokens = tokenizer.tokenize(curr_line)
        code_tokens = tokenizer.tokenize(curr_code)
        tokens = ([tokenizer.cls_token]
                  + line_tokens
                  + [tokenizer.sep_token]
                  + code_tokens
                  + [tokenizer.eos_token])
        
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN-1] + [tokenizer.eos_token]
        
        ids = tokenizer.convert_tokens_to_ids(tokens)
        attn_mask = [True] * MAX_SEQ_LEN
        
        if len(ids) < MAX_SEQ_LEN:
            pad_length = MAX_SEQ_LEN - len(ids)
            ids += [tokenizer.pad_token_id] * pad_length
            attn_mask[-pad_length:] = [False] * pad_length
            
        token_ids.append(ids)
        attn_masks.append(attn_mask)
        
    token_ids = torch.tensor(token_ids, device=device)
    attn_masks = torch.tensor(attn_masks, device=device)
        
    return token_ids, attn_masks

def baseline_accuracy(data_file_path: str)->float:
    data = CSVDataset(data_file_path)
    loader = DataLoader(data, batch_size=1)
    
    size = len(loader.dataset)
    correct = 0
    
    for _, _, label in loader:
        if label.item() == 0:
            correct += 1
    
    accuracy = correct / size        
    
    return accuracy

# Never used
def weighted_loss(y_pred, y):
    cross_fn = nn.CrossEntropyLoss()
    loss = cross_fn(y_pred, y)
    weights = torch.tensor([[2.0], [1.0]], device=device)
    loss += torch.matmul(y_pred, weights).sum()
    return loss

def convert_1d_label_to_2d(label: torch.Tensor)->torch.Tensor:
    labels = []
    for i in range(len(label)):
        if label[i]:
            labels.append([0, 1])
        else:
            labels.append([1, 0])
    labels = torch.tensor(labels, device=device, dtype=torch.float32)
    
    return labels

def eval(model: SideEffectClassificationModel, 
         tokenizer: AutoTokenizer, 
         eval_datafile_path: str, 
         batch_size: int, 
         loss_fn: Callable)->float:
    
    model.eval()
    
    eval_data = CSVDataset(eval_datafile_path)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    
    size = len(eval_loader.dataset)
    num_batches = len(eval_loader)
    eval_loss = 0
    correct = 0
    
    with torch.no_grad():
        for line, code, label in eval_loader:
            tokenized_batch, attn_masks = tokenize_batch((line, code), 
                                                         tokenizer)
            
            y_pred = model(tokenized_batch, attn_masks=attn_masks)
            labels = convert_1d_label_to_2d(label)
    
            eval_loss += loss_fn(y_pred, labels)
            correct += (y_pred.argmax(dim=1) == label).type(torch.float).sum()
            
    eval_loss /= num_batches
    accuracy = correct / size
    return eval_loss, accuracy


def train(args: argparse.Namespace)->None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    train_data = CSVDataset(args.train_data_file)
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True)
    
    config = RobertaConfig.from_pretrained(args.config_name)
    model = SideEffectClassificationModel(config).to(device=device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=args.scheduler_patience)
    
    model_dir = args.output_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    best_eval_loss = np.inf
    baseline_eval_accuracy = baseline_accuracy(args.eval_data_file)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()

        for batch, (line, code, label) in enumerate(train_loader):
            tokenized_batch, attn_masks = tokenize_batch((line, code), 
                                                         tokenizer)
            
            y_pred = model(tokenized_batch, attn_masks=attn_masks)
            labels = convert_1d_label_to_2d(label)
            
            loss = loss_fn(y_pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                print(f'Loss @ Epoch #{epoch} batch #{batch}: {loss}')
        
        print(f'Duration for epoch training: {time.time() - epoch_start_time}')
        
        eval_loss, accuracy = eval(model, 
                                   tokenizer, 
                                   args.eval_data_file, 
                                   args.batch_size, 
                                   loss_fn)
        print(f'Eval loss @ Epoch #{epoch}: {eval_loss}')
        print(f'Accuracy: {accuracy} vs Baseline: {baseline_eval_accuracy}')
        
        scheduler.step(eval_loss)
        
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), 
                       f'{model_dir}/{BEST_MODEL_WEIGHTS}')
            best_eval_loss = eval_loss
            
        torch.save(model.state_dict(), 
                   f'{model_dir}/model_weights_epoch_{epoch}.pth')
        
        print(f'Epoch duration: {time.time() - epoch_start_time}\n')


def test(args: argparse.Namespace)->None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    test_data = CSVDataset(args.test_data_file)
    test_loader = DataLoader(test_data, 
                              batch_size=args.batch_size, 
                              shuffle=True)
    
    config = RobertaConfig.from_pretrained(args.config_name)
    model = SideEffectClassificationModel(config).to(device=device)
    model_dir = args.output_dir
    model.load_state_dict(torch.load(f'{model_dir}/{BEST_MODEL_WEIGHTS}', 
                                     weights_only=True))
    loss_fn = nn.CrossEntropyLoss()
    
    _, accuracy = eval(model, 
                       tokenizer, 
                       args.test_data_file, 
                       args.batch_size, 
                       loss_fn)
    
    baseline_test_accuracy = baseline_accuracy(args.test_data_file)
    
    print(f'Test Accuracy: {accuracy} vs Baseline: {baseline_test_accuracy}')
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=DEFAULT_TRAIN_FILE, 
                        type=str, help="Train data file.")
    parser.add_argument("--eval_data_file", default=DEFAULT_EVAL_FILE, 
                        type=str, help="Evaluation data file.")
    parser.add_argument("--test_data_file", default=DEFAULT_TEST_FILE, 
                        type=str, help="Test data file.")
    
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str,
                        help="Where the model checkpoints will be written.")

    parser.add_argument("--config_name", default=DEFAULT_CONFIG_NAME, 
                        type=str, help="Pretrained config name.")
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER, 
                        type=str, help="Pretrained tokenizer name.")

    parser.add_argument("--do_train", default=DEFAULT_TO_TRAIN, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=DEFAULT_TO_TEST, type=bool,
                        help="Whether to run eval on the test set.")

    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, 
                        type=int, help="Batch size.")
    parser.add_argument("--learning_rate", default=DEFAULT_LEARNING_RATE, 
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=DEFAULT_EPSILON, 
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=DEFAULT_MAX_GRAD_NORM, 
                        type=float, help="Max gradient norm.")
    parser.add_argument("--scheduler_patience", default=SCHEDULER_PATIENCE, 
                        type=int, help="Patience for the lr scheduler.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs.")

    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="Random seed for initialization.")

    args = parser.parse_args()
    
    if args.do_train:
        train_path = args.train_data_file
        eval_path = args.eval_data_file
        if not all_paths_exist([train_path, eval_path]):
            raise Exception(f'Missing "{train_path}" & "{eval_path}".')
        
        train(args)
        
    elif args.do_test:
        test_path = args.test_data_file
        out_path = args.output_dir
        if not all_paths_exist([test_path, out_path]):
            raise Exception(f'Missing "{test_path}" or "{out_path}".')
        
        test(args)
            
if __name__ == '__main__':
    main()