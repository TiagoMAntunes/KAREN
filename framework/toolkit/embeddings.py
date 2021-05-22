# collection of pre-embeddings available for use inside the model
import os
from ..register_embeddings import RegisterEmbedding
from ..base_embedding import BaseEmbedding


import numpy as np

SAVEFOLDER = 'embeddings_data/'

if not os.path.exists(SAVEFOLDER):
    os.mkdir(SAVEFOLDER)


def download(url, savelocation):
    """ Downloads the file from that URL if it exists. Returns True if downloaded """
    import urllib
    import shutil

    if os.path.exists(savelocation):
        return False

    os.mkdir(savelocation)

    with urllib.request.urlopen(url) as response, open(savelocation + 'download.zip', 'wb') as f:
        print(f"Downloading file from {url}. This might take a while so you can monitor the download size in {savelocation + 'download.zip'}")
        shutil.copyfileobj(response, f)
        print('Download finished!')
    return True


@RegisterEmbedding('Glove')
class Glove(BaseEmbedding):
    """
        GloVe embeddings

        Source: https://github.com/stanfordnlp/GloVe
    """

    def process(URL, NAME):

        new = download(URL, NAME)
        if new:
            # need to unzip and etc
            import zipfile
            print('Extracting zip')
            with zipfile.ZipFile(NAME+'download.zip') as f:
                f.extractall(NAME)

            # delete
            os.remove(NAME + 'download.zip')

            # for efficient reading, we should transform all the files into binary format
            print('Transforming data into binary format')
            for name in tuple(filter(lambda x: x.endswith('.txt'), os.listdir(NAME))):
                filename = NAME + name
                vocab = []
                embeddings = []
                dimension = int(name.split('.')[3][:-1])
                with open(filename, encoding='utf-8') as f:
                    for line in f:
                        line = line.split()
                        if len(line) < dimension + 1:
                            # FIXME Some unicode character was giving trouble 
                            continue 
                        vocab.append(line[0])
                        embeddings.append(list(map(float, line[1:])))


                # save each one to .vocab and .embeddings
                embeddings = np.array(embeddings)
                with open(filename + '.vocab', 'w') as f:
                    f.write(' '.join(vocab))
                np.save(filename + '.embeddings', embeddings)

    @classmethod
    def get(cls, dim=200):
        URL = 'http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.twitter.27B.zip'
        NAME = SAVEFOLDER + 'glove_twitter/'
        cls.process(URL, NAME)

        files = {
            25: 'glove.twitter.27B.25d.txt',
            50: 'glove.twitter.27B.50d.txt',
            100: 'glove.twitter.27B.100d.txt',
            200: 'glove.twitter.27B.200d.txt',
        }

        if dim not in files:
            raise ValueError(
                f'Unavailable embedding dim {dim} for Glove Embeddings')
        
        with open(NAME + files[dim] + '.vocab') as f:
            vocab = f.read().split()
        
        return vocab, np.load(NAME + files[dim] + '.embeddings.npy')
