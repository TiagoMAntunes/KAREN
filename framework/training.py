import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
import random

def train(model, dataset, loss_fn, optimizer, max_iterations=30, seed=12345, split_amount=0.9, device="cpu", batch_size=256):
    def collate_fn(data):
        tensors, nontensors = dataset.__class__.get_properties()

        tensors_data = [x[:len(tensors)] for x in data]
        tensors_data = [[x[i] for x in tensors_data]
                        for i in range(len(tensors_data[0]))]

        nontensors_data = [x[len(tensors) + 1:] for x in data]
        nontensors_data = [[x[i] for x in nontensors_data]
                           for i in range(len(nontensors_data[0]))]

        data = {**{name: torch.tensor(x) for name, x in zip(tensors, tensors_data)}, **{
            name: x for name, x in zip(nontensors, nontensors_data)}}

        return data

    torch.manual_seed(seed - 1)
    np.random.seed(seed - 1)
    random.seed(seed - 1)

    # TODO early stopping

    splitter = lambda x: [round(split_amount*len(x)), len(x) - round(split_amount*len(x))]
    train, dev = torch.utils.data.random_split(dataset, splitter(dataset), generator=torch.Generator().manual_seed(seed))
    train, test = torch.utils.data.random_split(train, splitter(train), generator=torch.Generator().manual_seed(seed+1))

    train = torch.utils.data.DataLoader(
        train, batch_size=batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    dev = torch.utils.data.DataLoader(
        dev, batch_size=batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    test = torch.utils.data.DataLoader(
        test, batch_size=batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model.to(device)

    best_score = 0
    best_model = None
    display_freq = 10

    for iteration in range(max_iterations):

        totloss = 0
        c = 0

        model.train()
        with tqdm(train) as progress:
            for i, batch in enumerate(progress):
                for key in batch:
                    if not torch.is_tensor(batch[key]):
                        continue
                    batch[key] = batch[key].to(device)

                for param in model.parameters():
                    param.grad = None

                outputs = model(batch)
                loss = loss_fn(outputs, batch['label'])
                loss.backward()
                optimizer.step()

                totloss += loss.item()
                c += 1
                
                if i % display_freq == 0:
                    progress.set_postfix({'loss': totloss / (i + 1)})


        correct, tot = eval(model, dev, device)
        accuracy = correct / tot
        print(f'Epoch #{iteration + 1} validation accuracy = {accuracy:4f}')

        if accuracy > best_score:
            print(f'Accuracy increased from {best_score} to {accuracy}, saving model.')
            best_score = accuracy
            best_model = copy.deepcopy(model)
    

    correct ,tot = eval(best_model, test, device)
    print(f'\nTest accuracy: {correct / tot}')
        

        
def eval(model, test, device):
    model.eval()
    tot = 0
    correct = 0
    for i, batch in enumerate(test):
        for key in batch:
            if not torch.is_tensor(batch[key]):
                continue
            batch[key] = batch[key].to(device)

        outputs = model(batch)
        results = outputs.argmax(dim=-1)

        correct += (results == batch['label']).sum()
        tot += outputs.shape[0]

    return correct, tot