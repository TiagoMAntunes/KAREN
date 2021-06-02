import framework.training
from framework.register_model import MODELS
from framework.register_dataset import DATASETS
from framework.register_embeddings import EMBEDDINGS

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from pprint import pprint


def add_model_params(parser):
    group = parser.add_argument_group()

    group.add_argument('--model', '-m', type=str,
                       help='Name of the model to run', nargs='+', required=True)
    group.add_argument('--lr', type=float,
                       help='The learning rate to be applied on the optimizer', default=1e-3)
    group.add_argument('--dropout', type=float,
                       help='The dropout to apply on the model', default=0.1)
    group.add_argument('--embeddings', type=str,
                       help='In case of using pretrained embeddings, the type to use', default=None)
    group.add_argument('--embedding-dim', type=int,
                       help='The size of the embeddings to use', default=200)


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
    group.add_argument('--batch-size', default=256, type=int,
                       help='Batch size used in the dataloaders')
    group.add_argument('--seed', default=12345, type=int,
                       help='Seed for reproducibility')


def get_specific_model_params(parser, args):
    group = parser.add_argument_group()

    # check if any model is not available
    invalid = [x for x in args.model if x not in MODELS]

    if invalid:
        print()
        print('Invalid model selection: {}. Available models:\n- {}'.format(invalid,
              "\n- ".join(sorted(MODELS.keys()))))
        raise ValueError('Invalid model selection')

    for m in args.model:
        MODELS[m].add_required_arguments(group)


def get_specific_dataset_params(parser, args):
    group = parser.add_argument_group()

    # check if any model is not available
    invalid = [x for x in args.dataset if x not in DATASETS]

    if invalid:
        print()
        print('Invalid dataset selection: {}. Available datasets:\n- {}'.format(invalid,
              "\n- ".join(sorted(DATASETS.keys()))))
        raise ValueError('Invalid dataset selection')

    # also check if the models can be run on the datasets
    for model, dataset in ((x, y) for x in args.model for y in args.dataset):
        r = set(MODELS[model].data_requirements())
        a = set(
            (lambda x, y: x+y)(*DATASETS[dataset].get_properties())
        )
        if not r.issubset(a):
            raise ValueError(
                f'Dataset {dataset} does not contain all model {model} requirements: {r.difference(a)} ')

    for d in args.dataset:
        DATASETS[d].add_required_arguments(group)


def get_embeddings(args):
    if not args.embeddings:
        return None, None

    name = args.embeddings.lower()

    if name not in EMBEDDINGS:
        raise ValueError(f'Framework does not support {args.embeddings}.')

    return EMBEDDINGS[name].get(dim=args.embedding_dim)


def start(args):
    # Create all the required data to perform the computation
    vocab, values = get_embeddings(args)
    if vocab:
        vocab = {j: i for i, j in enumerate(vocab)}

    for datasetname in args.dataset:
        d = DATASETS[datasetname].make_dataset(args)
        args.in_feat = d.get_input_feat_size()
        args.out_feat = d.get_output_feat_size()
        args.vocab_size = d.get_vocab_size()
        args.device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

        for modelname in args.model:
            print(f'\nStarting training of (Model={modelname} Dataset={datasetname})')

            if vocab:
                # pretrained embeddings, now needs to get the vocab list conversion
                words2idx = d.words_to_idx()
                # gets the correct order of the words that exist in the embeddings
                indices = [vocab[x] if x in vocab else vocab['<unk>']
                        for x in words2idx]
                args.embeddings = nn.Embedding.from_pretrained(torch.tensor(values[indices]))
            else:
                # not pretrained ones, generate default embeddings
                args.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)

            model = MODELS[modelname].make_model(args)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=7, gamma=0.1)

            framework.training.train(model, d, criterion, optimizer, scheduler, max_iterations=args.max_epochs,
                                     device=args.device, batch_size=args.batch_size, seed=args.seed)


def reproducible(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    reproducible(12345)

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
    args.model = [x.lower() for x in args.model]
    args.dataset = [x.lower() for x in args.dataset]
    get_specific_model_params(parser, args)
    get_specific_dataset_params(parser, args)

    # now parse them all
    args = parser.parse_args()
    args.model = [x.lower() for x in args.model]
    args.dataset = [x.lower() for x in args.dataset]

    # remove duplicated models and datasets
    args.model = list(set(args.model))
    args.dataset = list(set(args.dataset))

    print('*'*30, ' CONFIGURATION ', '*'*30)
    print('\n'.join([f'{k:40}{v}' for k, v in sorted(
        list(vars(args).items()), key=lambda x: x[0])]))
    print('*'*77, '\n')

    t = start(args)
