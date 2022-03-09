import pandas as pd
from os import listdir
from os.path import isfile, join



def get_dir_files(dir_path, endwith=None, verbose=0):
    fichiers = None
    if endwith is not None:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
    else:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return fichiers