from framework import *

if __name__ == '__main__':
    dataset = HateXPlain()
    len(dataset)
    # print(dataset.data)
    train(None, dataset, None, None)