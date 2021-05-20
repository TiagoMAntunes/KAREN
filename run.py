import framework.training

import torch
import torch.nn as nn
import argparse


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

    group.add_argument('--cpu', action='store_true', help='Whether to use CPU instead of CUDA to run the model')
    group.add_argument('--max-epoch', '-epochs', default=50, type=int, help='The number of iterations to run the optimization')

if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser()

    # model
    add_model_params(parser)
    # dataset
    add_dataset_params(parser)
    # iteration specific
    add_iteration_params(parser)

    args = parser.parse_known_args()

    # Now that we know how the model will run, we need to find out specific parameters for the model and dataset (if available)
    
    print(args)

    # dataset = HateXPlain()
    # len(dataset)

    # model = SoftmaxRegression(
    #     dataset.get_input_feat_size(), dataset.get_output_feat_size())
    # print(model)
    # train(model, dataset, nn.CrossEntropyLoss(),
    #       torch.optim.Adam(model.parameters()))
    dataset = framework.datasets.HateXPlain()
    model = framework.models.SoftmaxRegression
    # framework.training.train()