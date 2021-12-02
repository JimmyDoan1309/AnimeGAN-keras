import os
import shutil
import tarfile
from urllib.request import urlretrieve
from fire import Fire


DATASETS = {
    'Hayao': 'https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Hayao.tar.gz',
    'Paprika': 'https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Paprika.tar.gz',
    'Shinkai': 'https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Shinkai.tar.gz',
}
        

def _download(dataset, saved_path='./dataset'):
    assert dataset in DATASETS.keys(), f'{dataset} must in {DATASETS.keys()}'
    url = DATASETS[dataset]
    
    style_path = os.path.join(saved_path, dataset)
    
    if not os.path.exists(style_path):
        print(f'Download {d} dataset')
        os.makedirs(style_path)
        urlretrieve(url,f'./tmp/{dataset}.tar.gz')
        with tarfile.open(f'./tmp/{dataset}.tar.gz') as fp:
            fp.extractall(saved_path)
    

def run(datasets=None, saved_path='./dataset'):
    if datasets is None:
        datasets = list(DATASETS.keys())  
    elif isinstance(datasets, str):
        datasets = datasets.split(',')
    
    os.makedirs('./tmp')
    for d in datasets:
        _download(d)
    
    shutil.rmtree('./tmp')
    
if __name__ == '__main__':
    Fire(run)