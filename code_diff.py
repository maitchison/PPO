"""
Check two source folders, see if they match or not..
"""

import difflib, os, hashlib

def get_python_files(path:str):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.py']

def get_hash(path:str):
    with open(path, 'rt') as t:
        s = "\n".join(t.readlines())
    return hashlib.md5(s.encode('utf8')).hexdigest()

def get_folders(path:str):
    return [x[0] for x in os.walk(path)]

def hash_of_dict(d:dict):
    keys = sorted(list(d.keys()))
    s = "".join(d[k] for k in keys)
    return hashlib.md5(s.encode('utf8')).hexdigest()

def check_folder(folder):
    """
    Make sure all experiments share the same code.
    """

    folder_hashes = set()

    for folder in get_folders(folder):
        if os.path.split(folder)[-1] != "rl":
            continue
        files = get_python_files(folder)
        hashes = {os.path.split(file)[-1]: get_hash(file) for file in files}
        this_hash = hash_of_dict(hashes)
        if this_hash not in folder_hashes:
            print(f"[{this_hash[:8]}] {folder}")
            folder_hashes.add(this_hash)

check_folder("./Run/TVF_BETA1")
check_folder("./Run/TVF_BETA2")
check_folder("./Run/TVF_REF")
print("Done.")

