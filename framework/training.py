import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, dataset, loss_fn, optimizer, max_iterations=50, seed=12345):
    def collate_fn(data):
        tensors, nontensors = dataset.__class__.get_properties()

        tensors_data = [x[:len(tensors)] for x in data]
        tensors_data = [[x[i] for x in tensors_data] for i in range(len(tensors_data[0]))]

        nontensors_data = [x[len(tensors) + 1:] for x in data]
        nontensors_data = [[x[i] for x in nontensors_data] for i in range(len(nontensors_data[0]))]
        
        data = {**{name: torch.tensor(x) for name, x in zip(tensors, tensors_data)}, **{name: x for name, x in zip(nontensors, nontensors_data)}}

        return data

    # TODO early stopping, model saving?

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    train, test = torch.utils.data.random_split(dataset, [round(0.9*len(dataset)), len(
        dataset) - round(0.9*len(dataset))], generator=torch.Generator().manual_seed(seed))

    train = torch.utils.data.DataLoader(
        train, batch_size=256, num_workers=4, collate_fn=collate_fn)

    test = torch.utils.data.DataLoader(test, batch_size=256, num_workers=2)

    for iteration in range(max_iterations):
        for i, batch in tqdm(enumerate(train)):
            print(i,batch)
            break

        break
