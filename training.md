# Training Neural Networks

- Neural networks are used as an universal function approximator. For any function, we have desired input (e.g. image) and have a desired output (e.g. probability distribution). If a proper activation function is used we can train a neural network to approximate a true function given the training set.
- When a network is initialized it is naive, it doesn't know the mapping of input and outputs. It is trained by showing examples of `input-output` pair. Then network parameters (weights and biases) are adjusted to approximate this function.
- To fine tune the parameters we need to know how much the predictions are deviating from the desired output. A loss function (cost) can be calculatedd to estimate this, it represents the prediction error (e.g. RMSE for regression problems).
- Network predictions depends on it's parameters, and the cost/loss depends on the prediction of the network. There is an assosciation of cost/loss with the network parameters. If we minimize the loss with respect to network parameters, our network parameters can approach towards an optimal configurations for which the predicted outputs are accurate.
- To incremently update the network parameters, towards an optimal configuration, we use `Gradient Descent`. Network weights are updated wrt gradient of loss function (negative/downwards).
- Gradient is the slope of loss function wrt network parameters.
