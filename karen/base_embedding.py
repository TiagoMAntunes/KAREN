
class BaseEmbedding():
    """
        Interface to be implemented by all embedding classes
    """

    @classmethod
    def get(cls, *args):
        raise NotImplementedError
