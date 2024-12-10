import os

def all_paths_exist(paths: list[str])->bool:
    '''
    Check if all paths exist.
    
    Inputs: paths - Each elem should be a file/dir path
    
    Output: True if all paths exist, False otherwise
    '''
    for path in paths:
        if not os.path.exists(path):
            return False
    return True