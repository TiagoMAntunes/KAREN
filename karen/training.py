import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
import os


def train(
    model,
    dataset,
    loss_fn,
    optimizer,
    scheduler,
    max_iterations=30,
    seed=12345,
    split_amount=0.9,
    device="cpu",
    batch_size=256,
):

    requirements = set(model.data_requirements() + ["label"])

    def collate_fn(data):
        tensors, nontensors, _ = dataset.__class__.get_properties()

        tensors_data = [x[: len(tensors)] for x in data]
        tensors_data = [[x[i] for x in tensors_data] for i in range(len(tensors_data[0]))]

        nontensors_data = [x[len(tensors) :] for x in data]
        nontensors_data = [[x[i] for x in nontensors_data] for i in range(len(nontensors_data[0]))]

        # filter now the ones that will be used by the model
        data = {
            **{name: torch.tensor(x) for name, x in zip(tensors, tensors_data) if name in requirements},
            **{name: x for name, x in zip(nontensors, nontensors_data) if name in requirements},
            **{name: x for name, x in dataset.get_extra_properties().items() if name in requirements},
        }

        return data

    def _init_fn(i):
        # Reproducibility
        np.random.seed(seed + i)

    # TODO early stopping

    splitter = lambda x: [round(split_amount * len(x)), len(x) - round(split_amount * len(x))]
    train, dev = torch.utils.data.random_split(dataset, splitter(dataset))
    train, test = torch.utils.data.random_split(train, splitter(train))

    nworkers = os.cpu_count()

    train = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=nworkers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    dev = torch.utils.data.DataLoader(
        dev,
        batch_size=batch_size,
        num_workers=nworkers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    test = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        num_workers=nworkers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

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
                loss = loss_fn(outputs, batch["label"])
                loss.backward()
                optimizer.step()

                totloss += loss.item()
                c += 1

                if i % display_freq == 0:
                    progress.set_postfix({"loss": totloss / (i + 1), "lr": round(scheduler.get_last_lr()[0], 6)})

        scheduler.step()

        accuracy, scores = eval(model, dev, device)
        print(f"Epoch #{iteration + 1} validation accuracy = {accuracy:4f}")
        # pretty_print_score(scores, dataset.get_labels())

        if accuracy > best_score:
            print(f"Accuracy increased from {best_score} to {accuracy}, saving model.")
            best_score = accuracy
            best_model = copy.deepcopy(model)

    accuracy, scores = eval(best_model, test, device)
    print(f"\nTest accuracy: {accuracy}")
    pretty_print_score(scores, dataset.get_labels())


def pretty_print_score(scores, labels):
    rows = list(zip(labels, *scores))
    print(tabulate(rows, ["Label name", "Precision", "Recall", "F1", "Counts"]))


def eval(model, test, device):
    model.eval()
    guesses = []
    correct = []
    for i, batch in enumerate(test):
        for key in batch:
            if not torch.is_tensor(batch[key]):
                continue
            batch[key] = batch[key].to(device)

        outputs = model(batch)
        results = outputs.argmax(dim=-1)

        guesses.extend(results.cpu().numpy())
        correct.extend(batch["label"].cpu().numpy())

    accuracy = accuracy_score(correct, guesses)
    scores = precision_recall_fscore_support(correct, guesses)
    return accuracy, scores
