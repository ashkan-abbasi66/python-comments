# Deep Learning with PyTorch

* **Part 1:** Introduction to PyTorch and using tensors
* **Part 2:** Building fully-connected neural networks with PyTorch
* **Part 3:** How to train a fully-connected network with backpropagation on MNIST
* **Part 4:** Train a neural network (MLP) on Fashion-MNIST
* **Part 5:** Fashion-MNIST; Training, Validation, Inference. How to reduce overfitting through Early Stopping or Dropout.
  * [**Introduction to Dropout**](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
  * MNIST; [Basic MLP](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb); [MLP w/ dropout](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-dropout.ipynb); [MLP w/ BN](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-batchnorm.ipynb)
  * [An example in Keras](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-dropout-with-keras.md) - [Example 2 in Keras](./assets/Dropout_Example.pdf)
  * **[CIFAR10; Dropout with different drop rates in practice](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)**
* **Part 6:** How to save and load trained models
* **CNN:** CIFAR10 dataset; `./cnn-cifar/cifar10_cnn_solution.ipynb`
* **Part 7:** Load image data with `torchvision`, also data augmentation
* **CNN w/ data augmentation:** CIFAR10 dataset; `./cnn-cifar/cifar10_cnn_augmentation.ipynb`
  * test it w/o augmentation on the validation and test sets.
* **Part 8:** Use transfer learning to train a state-of-the-art image classifier for dogs and cats
* **Weight Initialization:** All zeros/ones; Uniform/Normal distributions; `./weight-initialization`
  * More on weight initialization: [link 1](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html#How-to-find-appropriate-initialization-values); [link 2](https://www.askpython.com/python-modules/initialize-model-weights-pytorch)
* **Batch normalization**: `batch-norm` folder; 
  * [rasbt/batchnorm](../rasbt-intro-to-DL/L11/code/batchnorm.ipynb)
  * [A simple implementation of Batch Normalization using pytorch.](https://github.com/Johann-Huber/batchnorm_pytorch) [copied in batch-norm folder]



# Notes

**Todo**

- [ ] new CNN examples `cifar10_cnn` and `cifar10_cnn_augmentation`
- [ ] Tokenization - TEXT analysis.
- [ ] 
- [x] CNN examples (MNIST; [Plain CNN](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-basic.ipynb); [CNN w/ He initialization](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-he-init.ipynb))
- [x] Implement different initialization methods in PyTorch 
- [ ] 



**[loss functions](./loss%20functions.ipynb)**: Use logits with `nn.CrossEntropyLoss`, and use `nn.LogSoftmax` with `nn.NLLLoss`. See also [`LogProbabilityAndLogSoftmax.pdf`](./assets/LogProbabilityAndLogSoftmax.pdf).

**[autograd](./autograd.ipynb)**

**[datasets](./datasets.ipynb)**

**[data manipulation layers](./data%20manipulation%20layers.ipynb)**: max pooling (MaxPool2d), batch normalization (BatchNorm1d), Dropout.

**[Tensorboard in PyTorch](./Tensorboard%20in%20PyTorch.ipynb)** - [video](https://www.youtube.com/watch?v=6CEld3hZgqc&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=5)

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

# Main Training Loop of your program
total_loss = []
global iteration

model.train()
for batch_idx, (inputs, targets) in enumerate(train_loader):
	optimizer.zero_grad()
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets)
    
    total_loss.append(loss.item())
    
    writer.add_scalar('train_loss_logs', loss.item(), iteration)
    
    iteration += 1
    
    loss.backward()
    optimizer.step()

epoch_loss = sum(total_loss)/len(total_loss)

```
Usage: ` tensorboard --logdir="./output/nodulemnist3d/220410_122404/Tensorboard_Results/"`


**Recurrent neural networks (RNNs) and long short-term memory (LSTM)**:
- [Chris Olah's LSTM post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Edwin Chen's LSTM post](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [Andrej Karpathy's blog post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Andrej Karpathy's lecture on RNNs and LSTMs from CS231n](https://www.youtube.com/watch?v=iX5V1WpxxkY)
- [MIT 6.S191 (2021): Recurrent Neural Networks](https://www.youtube.com/watch?v=qjrad0V0uJE)

**Convolutional neural networks (CNNs)**:
- [Kernels](https://setosa.io/ev/image-kernels/)
- [Justin Johnson's benchmarks for popular CNN models](https://github.com/jcjohnson/cnn-benchmarks)
- [A friendly introduction to Convolutional Neural Networks and Image Recognition](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)
- [MIT 6.S191 (2021): Convolutional Neural Networks](https://www.youtube.com/watch?v=AjtX1N_VT9E)

**Transfer Learning**:
- [Transfer Learning (C3W2L07) by Andrew Ng](https://www.youtube.com/watch?v=yofjFQddwHE)

**Optimization**:
  - [Unstable gradient problem (Vanishing / Exploding gradients)](http://neuralnetworksanddeeplearning.com/chap5.html)

