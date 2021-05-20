import framework.training
from framework.register_model import MODELS
from framework.register_dataset import DATASETS


import torch
import torch.nn as nn
import argparse
from pprint import pprint


def add_model_params(parser):
    group = parser.add_argument_group()

    group.add_argument('--model', '-m', type=str,
                       help='Name of the model to run', nargs='+', required=True)
    group.add_argument('--criterion', type=str,
                       help='The loss function to be used in the model (default: CrossEntropyLoss)', default='CrossEntropyLoss')
    group.add_argument('--optimizer', type=str,
                       help='The optimization method to run on the model (default: Adam)', default='Adam')


def add_dataset_params(parser):
    group = parser.add_argument_group()

    group.add_argument('--dataset', '-d', type=str,
                       help='Name of the dataset in which to train and evaluate the model', nargs='+', required=True)


def add_iteration_params(parser):
    group = parser.add_argument_group()

    group.add_argument('--cpu', action='store_true',
                       help='Whether to use CPU instead of CUDA to run the model')
    group.add_argument('--max-epochs', '--epochs', default=50, type=int,
                       help='The number of iterations to run the optimization')


def get_specific_model_params(parser, args):
    group = parser.add_argument_group()

    # check if any model is not available
    invalid = [x for x in args.model if x not in MODELS]

    if invalid:
        print()
        print('Invalid model selection: {}. Available models:\n- {}'.format(invalid, "\n- ".join(sorted(MODELS.keys()))))
        raise ValueError('Invalid model selection')

    for m in args.model:
        MODELS[m].add_required_arguments(group)

def get_specific_dataset_params(parser, args):
    group = parser.add_argument_group()

    # check if any model is not available
    invalid = [x for x in args.dataset if x not in DATASETS]

    if invalid:
        print()
        print('Invalid dataset selection: {}. Available datasets:\n- {}'.format(invalid, "\n- ".join(sorted(DATASETS.keys()))))
        raise ValueError('Invalid dataset selection')

    # also check if the models can be run on the datasets
    for model, dataset in ((x,y) for x in args.model for y in args.dataset):
        r = set(MODELS[model].data_requirements())
        a = set(
            ( lambda x,y: x+y ) (*DATASETS[dataset].get_properties())
        )
        if not r.issubset(a):
            raise ValueError(f'Dataset {dataset} does not contain all model {model} requirements: {r.difference(a)} ')


    for d in args.dataset:
        DATASETS[d].add_required_arguments(group)


def start(args):
    # Create all the required data to perform the computation
    datasets = [DATASETS[x].make_dataset(args) for x in args.dataset]
    for d in datasets:
        args.in_feat = d.get_input_feat_size()
        args.out_feat = d.get_output_feat_size()
        models = [MODELS[x].make_model(args) for x in args.model]

        # TODO criterion, optimizer
        print(args)
        for m in models:
            framework.training.train(m, d, nn.CrossEntropyLoss(), torch.optim.Adam(m.parameters()), max_iterations=args.max_epochs)

if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser()

    # model
    add_model_params(parser)
    # dataset
    add_dataset_params(parser)
    # iteration specific
    add_iteration_params(parser)

    args, _ = parser.parse_known_args()

    # Now that we know how the model will run, we need to find out specific parameters for the model and dataset (if available)

    get_specific_model_params(parser, args)
    get_specific_dataset_params(parser, args)

    # now parse them all
    args = parser.parse_args()
    
    # TODO display a resume of the configuration that will be run

    start(args)
