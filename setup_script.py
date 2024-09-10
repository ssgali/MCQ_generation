import os
import tarfile
import requests

url = "https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz"
file_name = "s2v_reddit_2015_md.tar.gz"

def download_file(url, file_name):
    response = requests.get(url)
    response.raise_for_status() 
    with open(file_name, 'wb') as f:
        f.write(response.content)

def extract_tar(file_name):
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall()  

if __name__ == "__main__":
    download_file(url, file_name)
    extract_tar(file_name)
    os.remove(file_name)  