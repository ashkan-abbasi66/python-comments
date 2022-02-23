# Deep Learning with PyTorch

This repo contains notebooks and related code for Udacity's Deep Learning with PyTorch lesson. This lesson appears in our [AI Programming with Python Nanodegree program](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

* **Part 1:** Introduction to PyTorch and using tensors
* **Part 2:** Building fully-connected neural networks with PyTorch
* **Part 3:** How to train a fully-connected network with backpropagation on MNIST
* **Part 4:** Train a neural network (MLP) on Fashion-MNIST
* **Part 5:** Fashion-MNIST; Training, Validation, Inference. How to reduce overfitting through Early Stopping or Dropout.
  * MNIST; [Basic MLP](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb); [MLP w/ dropout](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-dropout.ipynb); [MLP w/ BN](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-batchnorm.ipynb)
  * [An example in Keras](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-dropout-with-keras.md) - [Example 2 in Keras](./assets/Dropout_Example.pdf)
  * **[CIFAR10; Dropout with different drop rates in practice](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)**
* **Part 6:** How to save and load trained models
* **Part 7:** Load image data with `torchvision`, also data augmentation
* **Part 8:** Use transfer learning to train a state-of-the-art image classifier for dogs and cats





# Notes

Todo

- [x] CNN examples (MNIST; [Plain CNN](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-basic.ipynb); [CNN w/ He initialization](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-he-init.ipynb))
- [ ] Implement different initialization methods in PyTorch ([link 1](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html#How-to-find-appropriate-initialization-values); [link 2](https://www.askpython.com/python-modules/initialize-model-weights-pytorch))
- [ ] 



[loss functions](./loss%20functions.ipynb): Use logits with `nn.CrossEntropyLoss`, and use `nn.LogSoftmax` with `nn.NLLLoss`. See also [`LogProbabilityAndLogSoftmax.pdf`](./assets/LogProbabilityAndLogSoftmax.pdf).

[autograd](./autograd.ipynb)

[datasets](./datasets.ipynb)

[data manipulation layers](./data%20manipulation%20layers.ipynb): max pooling (MaxPool2d), batch normalization (BatchNorm1d), Dropout.

[Tensorboard in PyTorch](./Tensorboard%20in%20PyTorch.ipynb) - [video](https://www.youtube.com/watch?v=6CEld3hZgqc&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=5)



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
