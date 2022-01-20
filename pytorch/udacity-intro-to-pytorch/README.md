# Deep Learning with PyTorch

This repo contains notebooks and related code for Udacity's Deep Learning with PyTorch lesson. This lesson appears in our [AI Programming with Python Nanodegree program](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

* **Part 1:** Introduction to PyTorch and using tensors
* **Part 2:** Building fully-connected neural networks with PyTorch
* **Part 3:** How to train a fully-connected network with backpropagation on MNIST
* **Part 4:** Exercise - train a neural network on Fashion-MNIST
* **Part 5:** Using a trained network for making predictions and validating networks
* **Part 6:** How to save and load trained models
* **Part 7:** Load image data with torchvision, also data augmentation
* **Part 8:** Use transfer learning to train a state-of-the-art image classifier for dogs and cats



# Notes

[loss functions](./loss%20functions.ipynb): Use logits with `nn.CrossEntropyLoss`, and use `nn.LogSoftmax` with `nn.NLLLoss`.

[autograd](./autograd.ipynb): 

batch normalization:

- `batch-norm` folder

- [rasbt/batchnorm](../rasbt-intro-to-DL/L11/code/batchnorm.ipynb)

- [A simple implementation of Batch Normalization using pytorch.](https://github.com/Johann-Huber/batchnorm_pytorch) [copied in batch-norm folder]

Recurrent neural networks (RNNs) and long short-term memory (LSTM):
- [Chris Olah's LSTM post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Edwin Chen's LSTM post](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [Andrej Karpathy's blog post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Andrej Karpathy's lecture on RNNs and LSTMs from CS231n](https://www.youtube.com/watch?v=iX5V1WpxxkY)

Convolutional neural networks (CNNs):
- Basics of convolution:
  - [Kernels](https://setosa.io/ev/image-kernels/)
- Benchmark different architectures:
  - [Justin Johnson's benchmarks for popular CNN models](https://github.com/jcjohnson/cnn-benchmarks)

Optimization:
  - [Unstable gradient problem (Vanishing / Exploding gradients)](http://neuralnetworksanddeeplearning.com/chap5.html)



# Tools

PyTorch:

- [PyTorch Tutorials](https://pytorch.org/tutorials/)

Keras:

- [Introduction to Keras for engineers](https://keras.io/getting_started/intro_to_keras_for_engineers).
- [Introduction to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers)
- [Official Code examples](https://keras.io/examples/)
