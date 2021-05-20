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
    group.add_argument('--max-epoch', '-epochs', default=50, type=int,
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

    # TODO at the moment, datasets don't have parameters, do they need?



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
    print(args)
    # framework.training.train()
