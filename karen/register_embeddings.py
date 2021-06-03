from .base_embedding import BaseEmbedding

EMBEDDINGS = {}


def RegisterEmbedding(name):
    """
    Decorator to keep track of the embeddings avaible for usage on the framework

    @RegisterEmbedding('Glove')
    class Glove(BaseEmbedding):
        (...)
    """
    name = name.lower()

    def register(cls):
        if name in EMBEDDINGS:
            raise ValueError(f"Duplicate registry of model {name}")
        if not issubclass(cls, BaseEmbedding):
            raise ValueError(f"All models should be an extension of {BaseEmbedding.__name__}")
        EMBEDDINGS[name] = cls
        # print(f'Registered {cls}')
        return cls

    return register
