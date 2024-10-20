import os
def fast_scandir(dir_name):
    #https://stackoverflow.com/questions/18394147/how-to-do
    # -a-recursive-sub-folder-search-and-return-files-in-a-list/59803793#59803793
    subfolders= [f.path for f in os.scandir(os.path.normpath(dir_name)) if f.is_dir()]
    for dir_name in list(subfolders):
        subfolders.extend(fast_scandir(dir_name))
    return subfolders

def run_fast_scandir(dir_name, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir_name):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)


    for dir_name in list(subfolders):
        sf, f = run_fast_scandir(dir_name, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files
