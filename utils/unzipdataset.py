
import os, sys, tarfile
def extract_iamdataset_file(tar_url, extract_path='./datasets/words/'):
    print(tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        print(item)
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

#extract(r'IAM_Words\words.tgz',extract_path)
