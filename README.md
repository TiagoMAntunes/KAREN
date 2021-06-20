# KAREN: Unifying Hatespeech Detection and Benchmarking

This project started as a course project for the 2021 Natural Language Processing course at Tsinghua University and is still a work in progress. Our final project report is available in [report.pdf](report.pdf)
Contributions are accepted for further work.

## Introduction

Hate speech, also known as offensive or abusive language, is defined as *“any form of communication that disparages a person or group on the basis of some characteristic such as race, color ethnicity, gender, sexual orientation, nationality, religion or other characteristic”* (Nockleby, 2000). Nowadays, thanks to the availability of the internet and the emergence of social media, people have the tools necessary to express their opinions online. This of course includes the widespread dissemination of hate speech. Such speech has the potential of causing severe psychological consequences to individuals, or potentially promote verbal or even physical violence against a group. Due to these unwanted consequences, both the industry and academia have been working hard to develop techniques that can accurately detect such forms of hate. Such solutions, however, are not unified. Most research proposes a solution together with their own dataset and evaluates only on this dataset. This suffers from several problems.

Firstly, *bias*. Due to cultural differences and even just different points of view between different individuals, perception of hate speech varies and is very subjective, which will result in some datasets being especially biased on way or another.

Secondly, *dataset incompatibility*. It is common for some recent models to make use of metadata which can help improve results with the help of some background information, and this will often lead to a low compatibility between models and datasets.

Overall, it is hard to specify what is the current state of the art and what are the most promising research directions. Very few models can be directly compared as they are trained on different datasets.

To combat these issues we propose **KAREN**, a framework that intends to unify this research area. Our contribution provides an easy to use system that unifies the testing platform and can be easily utilised by beginners and researchers at the forefront of the field alike. It eases the design of data pre-processing and model implementation, allowing researchers to compare models themselves on their machines, or to contribute with their own datasets, meaning it is easily to get results on new research, compare with other baselines and test the durability of different models in different environments.

## Running

To run the framework, you just need to run the `run.py` file available at the root of the repository. To get started simply run:

```
python3 run.py --model softmaxregression --dataset hatexplain --dropout 0.15 --max-epochs 5
```

You can check the parameters of each model in its file or by checking the initial configuration when running it.

## Contributing

You can contribute to the framework by adding models and datasets that fit the format of the framework.
Please note that for simplification, we assumed this task as being a multi-class classification, so the model must output probabilities of `out_feat` size which will then be passed to a `softmax` function.

### Models

All implemented models must extend the superclass `BaseModel` in `framework/models/base_model.py` and override its methods (which will be used in the remaining training and testing scripts. You can see an example of a Softmax classification in `framework/models/softmax_regression.py`.

If your model requires specific arguments, you can request them from the parser using the `add_required_arguments(parser)` method. **At the moment, if you run multiple models with the same requirements it will not run**. You should also create a `make_model` function that picks up the arguments from the parser and extracts the one your model needs.

After implementing your model, you can add it to the framework by adding the `@RegisterModel` decorator. This will make sure the framework can find your model.

You'll also need to add an import in `framework/models/__init__.py`

**Note:** different models make use of different data and this framework intends to provide a unified way of testing them and easing implementation. There is a collection of requirements for each model to run that must be containted within the dataset. Please make sure that you're not repeating words, typos or writing them in a different way. You can check the available features of a dataset by checking their `data_requirements()` method.

#### Available arguments

When developing a model, some extra arguments are always available for selection. Currently, the list is the following:

- in_feat
- out_feat
- vocab_size
- device

The `make_model` function should refrain from using any others than this list and the arguments specified on `add_arguments` of itself.

### Datasets

Datasets are implemented similar to models. You must extend `BaseDataset` from the file `framework/datasets/base_dataset.py` and implemented the required logic. `framework/datasets/hatexplain.py` provides an example on how to implement a dataset with lazy preprocessing.

For registering datasets, you must use the `@RegisterDataset` decorator and add the import in the `framework/datasets/__init__.py`. All the remaining logic is the same as for the models.


## Results
The results are available in [results.md](results.md)
