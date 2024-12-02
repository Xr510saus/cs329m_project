import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import tree_sitter_cpp as tscpp

from main import tokenize_batch
from tree_sitter import Language, Parser
from notebooks.utils import all_paths_exist
from model import SideEffectClassificationModel
from transformers import AutoTokenizer, RobertaConfig


CODE_DIR_NAME = 'CPP_Files'
LABEL_DIR_NAME = 'Labels'
LINE_DATASET = 'line_dataset'
CUTOFF_DATASET = 'cutoff_dataset'
AST_DATASET = 'ast_dataset'
AST_NO_CODE_DATASET = 'ast_no_code_dataset'

SET_NAME = 'ASTNode'
DESIRED_INPUT_TYPE_DATASET = LINE_DATASET

DEFAULT_CONFIG_NAME = 'microsoft/graphcodebert-base'
DEFAULT_TOKENIZER = DEFAULT_CONFIG_NAME

BEST_MODEL_WEIGHTS = 'best_model_weights.pth'
COL_NAMES = ['line', 'label']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_weights(config_name: str,
                       model_weights_path: str)->SideEffectClassificationModel:
    
    config = RobertaConfig.from_pretrained(config_name)
    model = SideEffectClassificationModel(config).to(device)
    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    
    return model


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_set_name", default=SET_NAME,
                        type=str, help="Dataset under dev_test_set/")
    parser.add_argument("--desired_input_type_dataset", 
                        default=DESIRED_INPUT_TYPE_DATASET,
                        type=str, help="Which model to use for testing.")
    parser.add_argument("--config_name", default=DEFAULT_CONFIG_NAME, 
                        type=str, help="Pretrained config name.")
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER, 
                        type=str, help="Pretrained tokenizer name.")
    
    args = parser.parse_args()
    
    test_set_name = args.test_set_name
    test_set_dir = f'dev_test_set/{test_set_name}'
    code_dir = f'{test_set_dir}/{CODE_DIR_NAME}'
    label_dir = f'{test_set_dir}/{LABEL_DIR_NAME}'
    
    if not all_paths_exist([code_dir, label_dir]):
        err = f'Missing "{code_dir}" or "{label_dir}".'
        raise Exception(err)

    file_name = ''
    extension_tracker = dict()

    for _, _, files in os.walk(code_dir):
        for file in files:
            try:
                if file[-3:] == '.cc':
                    name = file[:-3]
                    file_name = name
                    extension_tracker[name] = 'cc'
                elif file[-4:] == '.cpp':
                    name = file[:-4]
                    file_name = name
                    extension_tracker[name] = 'cpp'
            except:
                # Exceptions likely occur due to the filename being less than 
                # 3/4 chars long, so we can skip since they cannot be the code 
                # files we're looking for.
                continue
            
    code_file_name = f'{code_dir}/{file_name}.{extension_tracker[file_name]}'
    label_file_name = f'{label_dir}/{file_name}.txt'
    
    if not all_paths_exist([code_file_name, label_file_name]):
        err = f'Missing "{code_file_name}" or "{label_file_name}".'
        raise Exception(err)
    
    input_type = args.desired_input_type_dataset
    model_dir_name = f'{input_type}_saved_models'
    model_weights_path = f'{model_dir_name}/{BEST_MODEL_WEIGHTS}'
    if not all_paths_exist([model_weights_path]):
        err = f'Missing "{model_weights_path}".'
        raise Exception(err)
    model = load_model_weights(args.config_name,
                                model_weights_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    lang_parser = Parser(Language(tscpp.language()))
    
    with (open(code_file_name, 'r') as code_file, 
          open(label_file_name, 'r') as label_file,
          torch.no_grad()):
        code = code_file.read().split('\n')
        labels = label_file.read().split('\n')
        
        assert(len(code) == len(labels))
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        labeled_segments = []
        
        for i in range(len(code)):
            if input_type == AST_DATASET:
                stripped_code = [line.strip() for line in code[:i+1]]
                tree = lang_parser.parse(bytes('\n'.join(stripped_code),
                                               encoding='utf-8'))
                curr_line = str(tree.root_node)
                curr_code = '\t'.join(stripped_code)
            
            elif input_type == AST_NO_CODE_DATASET:
                stripped_code = [line.strip() for line in code[:i+1]]
                tree = lang_parser.parse(bytes('\n'.join(stripped_code), 
                                               encoding='utf-8'))
                curr_line = str(tree.root_node)
                curr_code = 'na'
            
            elif input_type == CUTOFF_DATASET:
                curr_line = 'Side effect free?'
                stripped_code = [line.strip() for line in code[:i+1]]
                curr_code = '\t'.join(stripped_code)
            
            elif input_type == LINE_DATASET:
                curr_line = f'Line {i+1}: {code[i].strip()}'
                stripped_code = [line.strip() for line in code]
                curr_code = '\t'.join(stripped_code)
            
            else:
                err = f'Unknown input dataset type: {input_type}'
                raise Exception(err)
            
            (tokenized_batch, 
            attn_masks) = tokenize_batch(([curr_line], [curr_code]), tokenizer)
            
            y_pred = model(tokenized_batch, attn_masks=attn_masks)
            y_pred_label = y_pred.argmax().item()

            curr_label = int(labels[i])
            if y_pred_label == 1 and curr_label == 1:
                tp += 1
            elif y_pred_label == 1 and curr_label == 0:
                fp += 1
            elif y_pred_label == 0 and curr_label == 1 :
                fn += 1
            else:
                tn += 1
            
            segment = [code[i], y_pred_label]
            labeled_segments.append(segment)
            
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        try:
            precision = tp / (tp + fp)
        except:
            precision = np.inf
        try:
            recall = tp / (tp + fn)
        except:
            recall = np.inf
        f1 = 2 * precision * recall / (precision + recall)
        
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
            
    
    
    labeled_segments = np.array(labeled_segments)
    df = pd.DataFrame(labeled_segments, columns=COL_NAMES)
    csv_path = f'{test_set_dir}/{test_set_name}_{input_type}_predicted.csv'
    df.to_csv(csv_path, index=False)
    
    predicted_labels = np.array(labeled_segments[:, 1], dtype=np.int32)
    label_list = dict()
    label_list['labels'] = predicted_labels.tolist()
    json_path = f'{test_set_dir}/{test_set_name}_{input_type}_predicted.json'
    with open(json_path, 'w') as label_list_json:
        json.dump(label_list, label_list_json)

if __name__ == '__main__':
    main()