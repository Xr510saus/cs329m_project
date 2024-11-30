import os
import time
import argparse
import numpy as np
import pandas as pd
import tree_sitter as ts
import tree_sitter_cpp as tscpp

from main import test
from tree_sitter import Language, Parser
from notebooks.utils import all_paths_exist


CODE_DIR_NAME = 'split_cpps'
LABEL_DIR_NAME = 'split_labels'
LINE_DATASET = 'line_dataset'
CUTOFF_DATASET = 'cutoff_dataset'
AST_DATASET = 'ast_dataset'
AST_NO_CODE_DATASET = 'ast_no_code_dataset'
TEST_DATASET = 'test_dataset'
COL_NAMES = ['line', 'code', 'label']

SET_NAME = 'register_core_types'
DESIRED_INPUT_TYPE_DATASET = LINE_DATASET

DEFAULT_CONFIG_NAME = 'microsoft/graphcodebert-base'
DEFAULT_TOKENIZER = DEFAULT_CONFIG_NAME
DEFAULT_BATCH_SIZE = 16 # empirically found to be most time efficient


def create_data_files(input_type: str,
                      code_dir: str,
                      label_dir: str,
                      test_data_file: str,
                      file_names: list[str],
                      extension_tracker: dict[str, str],
                      parser: ts.Parser)->None:
    container = []

    for name in file_names:
        cpp_file = f'{code_dir}/{name}.{extension_tracker[name]}'
        txt_file = f'{label_dir}/{name}.txt'
        
        if not all_paths_exist([cpp_file, txt_file]):
            raise Exception(f'Could not find {cpp_file} and {txt_file}')
        
        with (open(cpp_file, 'r') as code, 
            open(txt_file, 'r') as labels):
            code_lines = code.read().splitlines()
            label_lines = labels.readlines()
            
            if len(code_lines) != len(label_lines):
                raise Exception(f'Length mismatch for {name}.')
            
            for i in range(len(code_lines)):
                curr_label = int(label_lines[i])
                
                if input_type == LINE_DATASET:
                    curr_line = f'Line {i+1}: {code_lines[i].strip()}'
                    stripped_code = [line.strip() for line in code_lines]
                    curr_code_block = '\t'.join(stripped_code)
                    
                elif input_type == CUTOFF_DATASET:
                    curr_line = 'Side effect free?'
                    stripped_code = [line.strip() for line in code_lines[:i+1]]
                    curr_code_block = '\t'.join(stripped_code)
                    
                elif input_type == AST_DATASET:
                    stripped_code = [line.strip() for line in code_lines[:i+1]]
                    tree = parser.parse(bytes('\n'.join(stripped_code), 
                                              encoding='utf-8'))
                    curr_line = str(tree.root_node)
                    curr_code_block = '\t'.join(stripped_code)
                    
                elif input_type == AST_NO_CODE_DATASET:
                    stripped_code = [line.strip() for line in code_lines[:i+1]]
                    tree = parser.parse(bytes('\n'.join(stripped_code), 
                                              encoding='utf-8'))
                    curr_line = str(tree.root_node)
                    curr_code_block = 'na'
                    
                else:
                    raise Exception(f'Dataset name "{input_type}" unknown.')
                
                write_str = [curr_line, curr_code_block, curr_label]
                container.append(write_str)
      
    container = np.array(container)
    
    df = pd.DataFrame(container, columns=COL_NAMES)
    df.to_csv(test_data_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set_name", default=SET_NAME,
                        type=str, help="Dataset under dev_test_set/")
    parser.add_argument("--code_dir_name", default=CODE_DIR_NAME,
                        type=str, help="Dir containaing pre-split code files.")
    parser.add_argument("--label_dir_name", default=LABEL_DIR_NAME,
                        type=str, help="Dir containaing pre-split labels.")
    parser.add_argument("--desired_input_type_dataset", 
                        default=DESIRED_INPUT_TYPE_DATASET,
                        type=str, help="Which model to use for testing.")
    args = parser.parse_args()
    
    proj_dir = os.path.dirname(os.path.realpath('__file__'))
    test_set_dir = f'{proj_dir}/dev_test_set/{args.test_set_name}'
    code_dir = f'{test_set_dir}/{args.code_dir_name}'
    label_dir = f'{test_set_dir}/{args.label_dir_name}'

    test_data_file = f'{test_set_dir}/test.csv'
    saved_model_dir = f'{args.desired_input_type_dataset}_saved_models'
    
    parser.add_argument("--test_data_file", default=test_data_file, 
                        type=str, help="Test data file.")
    parser.add_argument("--output_dir", default=saved_model_dir, type=str,
                        help="Where the model checkpoints will be written.")
    parser.add_argument("--config_name", default=DEFAULT_CONFIG_NAME, 
                        type=str, help="Pretrained config name.")
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER, 
                        type=str, help="Pretrained tokenizer name.")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, 
                        type=int, help="Batch size.")
    args = parser.parse_args()
    
    if not all_paths_exist([code_dir, label_dir]):
        err = f'Missing "{code_dir}" & "{label_dir}".'
        raise Exception(err)

    file_names = []
    extension_tracker = dict()

    for _, _, files in os.walk(code_dir):
        for file in files:
            try:
                if file[-3:] == '.cc':
                    name = file[:-3]
                    file_names.append(name)
                    extension_tracker[name] = 'cc'
                elif file[-4:] == '.cpp':
                    name = file[:-4]
                    file_names.append(name)
                    extension_tracker[name] = 'cpp'
            except:
                # Exceptions likely occur due to the filename being less than 
                # 3/4 chars long, so we can skip since they cannot be the code 
                # files we're looking for.
                continue

    lang_parser = Parser(Language(tscpp.language()))
    
    start_time = time.time()
    create_data_files(args.desired_input_type_dataset,
                      code_dir,
                      label_dir,
                      test_data_file,
                      file_names,
                      extension_tracker,
                      lang_parser)
    end_time = time.time()
    print(f'Time to finish creating test.csv: {end_time - start_time} s\n')
    
    test(args, test_set_dir)

if __name__ == '__main__':
    main()