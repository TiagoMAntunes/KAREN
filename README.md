# KAREN: Unifying Hatespeech Detection and Benchmarking

This project started as a course project for the 2021 Natural Language Processing course at Tsinghua University and is still a work in progress. Contributions are accepted for further work.

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


<<<<<<< HEAD
## Initial Results
HateXPlain evaluation. Precision, Recall and F1-score are the results from the hatespeech class.

| Model	| Accuracy	| Precision	| Recall | F1| 
| ------| ----------| ----------| -------| --| 
| **Bert**| **0.689**| 	**0.777**	| **0.752**| 	**0.764**|
| CNN	| 0.613	| 0.711	| 0.69	| 0.7| 
| Softmax Regression	| 0.366	| 0.298	| 0.087	| 0.135| 
| RNN	| 0.55	| 0.684| 	0.654| 	0.669| 
| BiLSTM| 	0.59| 	0.662| 	0.764| 	0.71| 
| NetLSTM	| 0.61| 	0.678| 	0.76| 	0.716| 
| GRU	| 0.609	| 0.666	| 0.777	| 0.72| 
| Transformer (1 layer)	| 0.486| 	0.495	| 0.6	| 0.543| 
| Transformer (2 layers)| 	0.532| 	0.551	| 0.732| 	0.629| 
| CharCNN | 0.552 | 0.65 | 0.61 | 0.63 |
| **AngryBERT (primary only)** | **0.649** | **0.736** | **0.695** | **0.764** |
| **DistilBERT** | **0.646** | **0.766** | **0.704** | **0.734** |
| RNN + GloVe	| 0.546| 	0.59	| 0.779	| 0.672| 
| **CNN + GloVe**	| **0.644**	| **0.69** | **0.767**| **0.726**| 
| **BiLSTM + GloVe**	| **0.637**| 	**0.677**	| **0.781**| 	**0.73**| 
| **GRU + GloVe**| **0.64**| 	**0.699**	| **0.736** | **0.717** | 
| NetLSTM + GloVe	| 0.616| 	0.679| 	0.756| 	0.715| 
| Transformer (1 layer) + GloVe	| 0.564	| 0.581	| 0.785	| 0.668| 
| Transformer (2 layers) + GloVe| 	0.572| 	0.751| 	0.609	| 0.672| 
| CharCNN + Glove | 0.573 | 0.631 | 0.753 | 0.686 | 
| **AngryBERT + Glove (primary only)** | **0.660** | **0.75** | **0.771** | **0.76** |
| UNet + Glove | 0.568 | 0.692 | 0.649 | 0.670 |
| UNet + Glove (using linear) | 0.553 | 0.624 | 0.631 | 0.628 |
| UNet | 0.560 | 0.634 | 0.659 | 0.620 |
| UNet (using linear) | 0.559 | 0.666 | 0.688 | 0.677 |
| VDCNN| 0.563 | 0.582633 | 0.776119 | 0.6656 |
| VDCNN + Glove 17 | 0.598 | 0.655 | 0.709 | 0.681 |
=======
## Results
The results are available in [results.md](results.md)
>>>>>>> master
