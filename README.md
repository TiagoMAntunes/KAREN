# Hate Speech Framework

## Introduction 
TODO

## Running

To run the framework, you just need to run the `run.py` file available at the root of the repository. To get started simply run:

```
python3 run.py --model softmaxregression --dataset hatexplain --dropout 0.15 --max-epochs 5
```


## Contributing

You can contribute to the framework by adding models and datasets that fit the format of the framework.

### Models

All implemented models must extend the superclass `BaseModel` in `framework/models/base_model.py` and override its methods (which will be used in the remaining training and testing scripts. You can see an example of a Softmax classification in `framework/models/softmax_regression.py`. 

If your model requires specific arguments, you can request them from the parser using the `add_required_arguments(parser)` method. **At the moment, if you run multiple models with the same requirements it will not run**. You should also create a `make_model` function that picks up the arguments from the parser and extracts the one your model needs.

After implementing your model, you can add it to the framework by adding the `@RegisterModel` decorator. This will make sure the framework can find your model.

You'll also need to add an import in `framework/models/__init__.py`

**Note:** different models make use of different data and this framework intends to provide a unified way of testing them and easing implementation. There is a collection of requirements for each model to run that must be containted within the dataset. Please make sure that you're not repeating words, typos or writing them in a different way. You can check the available features of a dataset by checking their `data_requirements()` method.

### Datasets

Datasets are implemented similar to models. You must extend `BaseDataset` from the file `framework/datasets/base_dataset.py` and implemented the required logic. `framework/datasets/hatexplain.py` provides an example on how to implement a dataset with lazy preprocessing. 

For registering datasets, you must use the `@RegisterDataset` decorator and add the import in the `framework/datasets/__init__.py`. All the remaining logic is the same as for the models.