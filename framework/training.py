import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, dataset, loss_fn, optimizer, max_iterations=30, seed=12345, split_amount=0.9, device="cpu"):
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

    # TODO early stopping, model saving?
    
    model.to(device)

    train, test = torch.utils.data.random_split(dataset, [round(split_amount*len(dataset)), len(
        dataset) - round(split_amount*len(dataset))], generator=torch.Generator().manual_seed(seed))

    train = torch.utils.data.DataLoader(
        train, batch_size=64, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    test = torch.utils.data.DataLoader(
        test, batch_size=64, num_workers=4, collate_fn=collate_fn, pin_memory=True)

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

    print(
        f'Before starting accuracy is {correct / tot} with {correct} correct out of {tot} total entries')

    for iteration in range(max_iterations):
        print('-'*5, f'Epoch {iteration+1}', '-'*5)

        totloss = 0
        c = 0
        model.train()
        for i, batch in tqdm(enumerate(train)):
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

        print(
            f'Finished! Eval accuracy = {correct / tot} with {correct} correct out of {tot} total entries, avg loss = {totloss / c}')
