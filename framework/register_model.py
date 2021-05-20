from .models import BaseModel

MODELS = {}


def RegisterModel(name):
    """
        Decorator to keep track of the models on the framework

        @RegisterModel('SoftmaxRegression')
        class SoftmaxRegression(BaseModel):
            (...)
    """
    name = name.lower()
    def register(cls):
        if name in MODELS:
            raise ValueError(f'Duplicate registry of model {name}')
        if not issubclass(cls, BaseModel):
            raise ValueError(
                f'All models should be an extension of {BaseModel.__name__}')
        MODELS[name] = cls
        print(f'Registered {cls}')
        return cls

    return register