import os

def all_paths_exist(paths: list[str])->bool:
    for path in paths:
        if not os.path.exists(path):
            return False
    return True