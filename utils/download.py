def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, url.split("/")[-1])

    if os.path.exists(file_name):
        print(f"Dataset already downloaded at {file_name}.")
    else:
        urllib.request.urlretrieve(url, file_name, ProgressBar().update)

    return 

def generate_data(source_folder, target_folder, target_class):

    txt_data = open(target_class, "r") 
    for ids, txt in enumerate(txt_data):
        s = str(txt.split('\n')[0])
        f.append(s)

    for ids, dirs in enumerate(os.listdir(source_folder)):
        for tg_class in f:
            if dirs == tg_class:
                print('{} is transferred'.format(dirs))
                shutil.copytree(os.path.join(source_folder,dirs), os.path.join(target_folder,dirs)) 