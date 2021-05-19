from framework import *

import torch, torch.nn as nn

if __name__ == '__main__':
    dataset = HateXPlain()
    len(dataset)
    
    model = Linear(dataset.get_input_feat_size(), dataset.get_output_feat_size())
    print(model)
    train(model, dataset, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()))