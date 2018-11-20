# Training Neural Networks

- Neural networks are used as an universal function approximator. For any function, we have desired input (e.g. image) and have a desired output (e.g. probability distribution). If a proper activation function is used we can train a neural network to approximate a true function given the training set.
- When a network is initialized it is naive, it doesn't know the mapping of input and outputs. It is trained by showing examples of `input-output` pair. Then network parameters (weights and biases) are adjusted to approximate this function.
- To fine tune the parameters we need to know how much the predictions are deviating from the desired output. A loss function (cost) can be calculatedd to estimate this, it represents the prediction error (e.g. RMSE for regression problems).
- Network predictions depends on it's parameters, and the cost/loss depends on the prediction of the network. There is an assosciation of cost/loss with the network parameters. If we minimize the loss with respect to network parameters, our network parameters can approach towards an optimal configurations for which the predicted outputs are accurate.
- To incremently update the network parameters, towards an optimal configuration, we use `Gradient Descent`. Network weights are updated wrt gradient of loss function (negative/downwards).
- Gradient is the slope of loss function wrt network parameters.
- Training multilayer neural networks is done through `backpropagation`, which is just an application of the calculus chain rule. For every forward pass, a gradient wrt loss function is calculated and backpropagated. Then weights are updated using these values.

## Losses in PyTorch

PyTprch loss functions are present in `nn` module (`nn.CrossEntropyLoss`). Losses are usually assigned to `criterion` variable. A loss function is assigned to a criterion variable and network output and the correct output is passed to the criterion.

Building the model with network logits (not probabilities)
```python
# build a feed forward network
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))
# define the loss
criterion = nn.CrossEntropy()
# get out data
images, labels = next(iter(trainloader))
# flattern image
images = images.view(images.shape[0], -1)
# forward pass and get our logits
logits = model(images)
# calculate the loss with logits and the labels
loss = criterion(logits, labels)
print(loss)
```