import os
import torch
import argparse
import pandas as pd
import torch.nn as nn

from transformers import AutoTokenizer, RobertaConfig
from model import SideEffectClassificationModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 5
DEFAULT_EPSILON = 1e-8
DEFAULT_SEED = 42
DEFAULT_DATASET = 'datasets/cutoff_dataset'
MAX_SEQ_LEN = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
proj_dir = os.path.dirname(os.path.abspath('__file__'))


class CSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        line = self.df.iloc[idx, 0]
        code = self.df.iloc[idx, 1]
        label = torch.tensor(self.df.iloc[idx, 2], device=device)
        
        return line, code, label
    
    
def tokenize_batch(batch: tuple, tokenizer: AutoTokenizer)->list[torch.Tensor]:
    token_ids = []
    
    for i in range(len(batch[0])):
        curr_line = batch[0][i]
        curr_code = batch[1][i]
        
        line_tokens = tokenizer.tokenize(curr_line)
        code_tokens = tokenizer.tokenize(curr_code)
        tokens = [tokenizer.cls_token]+line_tokens+[tokenizer.sep_token]+code_tokens #+[tokenizer.eos_token]
        
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN-1] + [tokenizer.eos_token]
        elif len(tokens) < MAX_SEQ_LEN:
            padding = MAX_SEQ_LEN - len(tokens)
            tokens += [tokenizer.eos_token] * padding
        else:
            tokens[-1] = tokenizer.eos_token
        
        token_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        
    return torch.tensor(token_ids, device=device)


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    train_data = CSVDataset(args.train_data_file)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    config = RobertaConfig.from_pretrained(args.config_name)
    model = SideEffectClassificationModel(config).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    
    model_dir = f'{proj_dir}/{args.output_dir}'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    for epoch in range(args.epochs):
        model.train()

        for batch, (line, code, label) in enumerate(train_loader):
            tokenized_batch = tokenize_batch((line, code), tokenizer)
            
            y_pred = model(tokenized_batch)
            loss = loss_fn(y_pred, label)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                print(f'Loss @ Epoch #{epoch+1} batch #{batch}: {loss}')
                
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'{model_dir}/model_weights_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), f'{model_dir}/latest_model_weights.pth')
    
    
def eval():
    return


def test():
    return
    
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=f'{DEFAULT_DATASET}/train.csv', type=str, #required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='saved_models', type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=f'{DEFAULT_DATASET}/eval.csv', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=f'{DEFAULT_DATASET}/test.csv', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=False, type=bool,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=DEFAULT_LEARNING_RATE, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=DEFAULT_EPSILON, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help="training epochs")

    args = parser.parse_args()
    
    if args.do_train:
        train(args)
    elif args.do_test:
        pass
            
if __name__ == '__main__':
    main()