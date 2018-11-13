# Intro

Tensors are the main datastructure of pytorch. pytorch uses autograd, which calculates gradient for each operation in the network, which further can be used to update the weights.

pytorch installation linux

```python
conda install pytorch torchvision cuda92 -c pytorch
```

pytorch installation windows

```python
conda install pytorch-cpu -c pytorch
pip3 install torchvision
```

## Single layer neural network

to an input we multiply weights and add a bias to pass it through an activation function to get an output. output is a linear combination of weights and biases matrix. A tensors are n dimensional matrices.

```python
sigmoid_fn(x) = 1 / (1 + torch.exp(-x))
```

while generating randomn numbers, we need to seed such that results can be reproduced consistently.

```python
torch.manual_seed(7)

#feature is 5 random normal variables
features = torch.randn((1, 5))
# creates a tensor with shape (1, 5), one row and five columns, that contains values randomly distributed according to the normal distribution with a mean of zero and standard deviation of one.

weights torch.randn_like(features)
# creates another tensor with the same shape as features, again containing values from a normal distribution.

biases = torch.randn((1, 1))
#creates a single value from a normal distribution
```
