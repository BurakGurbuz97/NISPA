from torchvision.datasets import  MNIST as MNIST_TORCH
from continuum.datasets import MNIST,  FashionMNIST, EMNIST, Fellowship, CIFAR100, CIFAR10
from continuum.datasets import InMemoryDataset
import torch
import numpy as np
from continuum import ClassIncremental


def get_dataset(dataset, increment, args = None):
    
    if dataset == 'mnist':
        train_dataset = MNIST('Data', train = True, download=True)
        test_dataset = MNIST('Data', train = False, download=True)
        input_dim = 28*28
     
    if dataset == 'fmnist':
        train_dataset = FashionMNIST('Data', train = True, download=True)
        test_dataset = FashionMNIST('Data', train = False, download=True)
        input_dim  = 28*28
    if dataset == 'emnist':
        train_dataset = EMNIST('Data', train = True, download=True, split='balanced')
        test_dataset = EMNIST('Data', train = False, download=True, split='balanced')
        input_dim  = 28*28

    if dataset=="cifar10":
        train_dataset = CIFAR10('Data', train = True, download=True)
        test_dataset = CIFAR10('Data', train = False, download=True)
        input_dim = 3

    if dataset == "cifar100":
        train_dataset = CIFAR100('Data', train = True, download=True)
        test_dataset = CIFAR100('Data', train = False, download=True)
        input_dim = 3
        
    if dataset == "pmnist_partial":
        return get_permuted_MNIST(args.perm_perc)

        
    if dataset == "emnist_fmnist":
        train_dataset = Fellowship([EMNIST('Data', train = True, download=True, split='balanced'), 
                                    FashionMNIST('Data', train = True, download=True)])
        
        test_dataset = Fellowship([EMNIST('Data', train = False, download=True, split='balanced'), 
                                   FashionMNIST('Data', train = False, download=True)])
        
        scenario_train = ClassIncremental(train_dataset, increment=[10, 13, 13, 11, 10])
        scenario_test = ClassIncremental(test_dataset, increment=[10, 13, 13, 11, 10])
        output_dim = max(train_dataset.get_data()[1]) + 1
        input_dim  = 28*28
        return scenario_train,  scenario_test, input_dim, output_dim
        
    output_dim = max(train_dataset.get_data()[1]) + 1
    
    scenario_train = ClassIncremental(train_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment)
    scenario_test = ClassIncremental(test_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment)
    
    
    return scenario_train, scenario_test, input_dim, output_dim


def partial_permutation(size, perc):
    indices = np.arange(0, size)
    if perc == 0:
        return indices, indices
    rand_indices = np.random.choice(indices, int(size / 100.0 * perc))
    perm = np.random.permutation(rand_indices)
    indices[rand_indices] = perm
    return np.arange(0, size), indices

#perm_percentage is list [0, 10, 20, ...]
#number of tasks is len(perm_percentage)
def get_permuted_MNIST(perm_percentage):
    mnist_train = MNIST_TORCH('Data', train = True, download=True)
    mnist_test = MNIST_TORCH('Data', train = False, download=True)
    
    x_train_original = mnist_train.data
    y_train = mnist_train.targets
    x_test_original = mnist_test.data
    y_test = mnist_test.targets
    
    x_train_all = []
    y_train_all = []
    x_test_all = []
    y_test_all = []
    
    for id_, perm_perc in enumerate(perm_percentage):
        x_train_flat = torch.flatten(x_train_original, start_dim = 1)
        x_test_flat = torch.flatten(x_test_original, start_dim = 1)
        old, new = partial_permutation(784, perm_perc)
        x_train_flat[:, old] = x_train_flat[:, new]
        x_test_flat[:, old] = x_test_flat[:, new]
        x_train_all.append(x_train_flat.reshape(-1, 28, 28))
        x_test_all.append(x_test_flat.reshape(-1, 28, 28))
        y_train_all.append(y_train + 10 * id_)
        y_test_all.append(y_test + 10 * id_)
        
    train_x = np.concatenate(x_train_all)
    test_x = np.concatenate(x_test_all)
    train_y = np.concatenate(y_train_all)
    test_y =  np.concatenate(y_test_all)
    train_dataset = InMemoryDataset(train_x, train_y)
    test_dataset = InMemoryDataset(test_x, test_y)
    
    scenario_train = ClassIncremental(train_dataset, increment=[10] * len(perm_percentage))
    scenario_test = ClassIncremental(test_dataset, increment=[10] * len(perm_percentage))
    return scenario_train, scenario_test, 784, 10 * len(perm_percentage)
    
    
    
    