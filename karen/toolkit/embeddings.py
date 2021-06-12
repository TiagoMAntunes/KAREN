# collection of pre-embeddings available for use inside the model
import os
from ..register_embeddings import RegisterEmbedding
from ..base_embedding import BaseEmbedding


import numpy as np

SAVEFOLDER = "embeddings_data/"

if not os.path.exists(SAVEFOLDER):
    os.mkdir(SAVEFOLDER)


def download(url, savelocation):
    """Downloads the file from that URL if it exists. Returns True if downloaded"""
    import urllib
    import wget

    if os.path.exists(savelocation):
        return False

    os.mkdir(savelocation)

    # with urllib.request.urlopen(url) as response, open(savelocation + "download.zip", "wb") as f:
    #     print(
    #         f"Downloading file from {url}. This might take a while so you can monitor the download size in {savelocation + 'download.zip'}"
    #     )
    #     wget.copyfileobj(response, f)
    #     print("Download finished!")
    print(f"Downloading file from {url}")
    wget.download(url, out=savelocation + "download.zip")
    print()
    return True

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

            print("Extracting zip")
            with zipfile.ZipFile(NAME + "download.zip") as f:
                f.extractall(NAME)

            # delete
            os.remove(NAME + "download.zip")

            # for efficient reading, we should transform all the files into binary format
            print("Transforming data into binary format")
            for name in tuple(filter(lambda x: x.endswith(".txt"), os.listdir(NAME))):
                filename = NAME + name
                embeddings = {}
                dimension = int(name.split(".")[-2][:-1])
                with open(filename, encoding="utf-8") as f:
                    for line in f:
                        line = line.split()
                        if len(line) < dimension + 1:
                            # FIXME Some unicode character was giving trouble
                            continue
                        embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)

                # save each one to .vocab and .embeddings
                with open(filename + ".vocab", "w") as f:
                    f.write(" ".join(embeddings.keys()))

                embeddings = np.array(list(embeddings.values()), dtype=np.float32)
                np.save(filename + ".embeddings", embeddings)
                del embeddings

    @classmethod
    def get(cls, **kwargs):
        cls.process(kwargs['URL'], kwargs['NAME'])

        files = kwargs['FILES']

        if kwargs['dim'] not in files:
            raise ValueError(f"Unavailable embedding dim {kwargs['dim']} for Glove Embeddings")

        with open(kwargs['NAME'] + files[kwargs['dim']] + ".vocab") as f:
            vocab = f.read().split()

        return vocab, np.load(kwargs['NAME'] + files[kwargs['dim']] + ".embeddings.npy")


@RegisterEmbedding('TwitterGlove')
class TwitterGlove(Glove):
    
    @classmethod
    def get(cls, dim=200):
        URL = "http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.twitter.27B.zip"
        NAME = SAVEFOLDER + "glove_twitter/"

        files = {
            25: "glove.twitter.27B.25d.txt",
            50: "glove.twitter.27B.50d.txt",
            100: "glove.twitter.27B.100d.txt",
            200: "glove.twitter.27B.200d.txt",
        }

        if dim not in files:
            raise ValueError(f'Dim {dim} is not a valid size for TwitterGlove. Available sizes: {[" ".join(files.keys())]}')

        return super(TwitterGlove, cls).get(URL=URL, NAME=NAME, FILES=files, dim=dim)


@RegisterEmbedding('CommonGlove')
class CommonGlove(Glove):

    @classmethod
    def get(cls, dim=300):
        URL = "https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"
        NAME = SAVEFOLDER + "glove_common/"

        files = {
            300: "glove.42B.300d.txt"
        }

        if dim not in files:
            raise ValueError(f'Dim {dim} is not a valid size for TwitterGlove. Available sizes: {[" ".join(files.keys())]}')

        return super(CommonGlove, cls).get(URL=URL, NAME=NAME, FILES=files, dim=dim)

@RegisterEmbedding('WikipediaGlove')
class WikipediaGlove(Glove):

    @classmethod
    def get(cls, dim=300):
        URL = 'https://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
        NAME = SAVEFOLDER + 'glove_wikipedia/'

        files = {
            50: "glove.6B.50d.txt",
            100: "glove.6B.100d.txt",
            200: "glove.6B.200d.txt",
            300: "glove.6B.300d.txt"
        }

        if dim not in files:
            raise ValueError(f'Dim {dim} is not a valid size for TwitterGlove. Available sizes: {[" ".join(files.keys())]}')

        return super(WikipediaGlove, cls).get(URL=URL, NAME=NAME, FILES=files, dim=dim)