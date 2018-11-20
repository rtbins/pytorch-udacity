# Intro

Tensors are the main datastructure of pytorch. Pytorch uses autograd which calculates gradient for each operation in the network, which further can be used to update the weights.

## Installation

Linux with gpu support

```python
conda install pytorch torchvision cuda92 -c pytorch
```

Windows

```python
conda install pytorch-cpu -c pytorch
pip3 install torchvision
```

## Single layer neural network

To an input a neural network multiply weights, add biases and pass it through an activation function to get an output (deterministic or stochastic). Output is a linear combination of weights and biases matrix. A tensors are n dimensional matrices.

```python
sigmoid_fn(x) = 1 ./ (1 + torch.exp(-x))
```

When generating randomn numbers, we need to seed such that results can be reproduced consistently when the setup is rerun.

```python
torch.manual_seed(7)

#feature is 5 random normal variables
features = torch.randn((1, 5))
'''
creates a tensor with shape (1, 5), one row and five columns,
that contains values randomly distributed according to the normal
distribution with a mean of zero and standard deviation of one.
'''
weights torch.randn_like(features) # 1 node in the hidden layer
'''
creates another tensor with the same shape as features, again
containing values from a normal distribution.
'''
biases = torch.randn((1, 1))
#creates a single value from a normal distribution
```

## Building neural networks with pytorch

PyTorch nn module simplifies building neural networks. Networks can be defined more concisely using `torch.nn.functional` module, these function represents simple element wise functions.

```python
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super().__init__()

    #all the layers can be defined as a part of constructor
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.hidden = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.output(x), dim=1)
```

## Activation functions

For a function to be used as activation function only requirement is non-linearity. The popular activation functions are

- Sigmoid (-inf, inf) => (0,1)
- tanh (-inf, inf) => (-1,1)
- relu (-inf, inf) => (0,inf)

## Using network class

```python
  model = Network()
  # print the weights, biases of a layer
  print(model.fc1.weght)
  print(model.fc1.bias)
  # Set biases to all zeros
  model.fc1.bias.data.fill_(0)
  # sample from random normal with standard dev = 0.01
  model.fc1.weight.data.normal_(std=0.01)
```

## nn.Sequential

PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, nn.Sequential

```python
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))

print(model[0])
model[0].weight
```

## Neural nets with `OrderDict`

```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
```
