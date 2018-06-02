# MNIST-Classification-using-Apache-MXNet

Apache MXNet is an open-source deep learning framework used to train and deploy deep neural networks. It is scalable, allowing for fast model training, and supports multiple languages (C++, Python, R, Scala, etc.). MXNet has two high-level interfaces: the Gluon API for imperative programming and the Module API for symbolic programming. We will focus on using the Gluon API for this tutorial. An imperative model works better for real-time/stochastic models because the graph is able to be modified over time.

First, we will use MXNet to perform linear regression with polynomial features  A deep learning library isn't needed to do this fairly simple task. However, working through this problem will introduce us to the mechanics of MXNet and give us a crash course on how to use the library.

In our second task, we will build a deep neural network called a multilayer perceptron (MLP) to classify hand-written digits. We will use the MNIST dataset, which is a collection of 28x28 greyscale images of handwritten digits and their correct labels.

The model we will build is purposely similar to the one we built for linear regression to show the power of abstraction using a deep learning library like Apache MXNet affords us. The main difference is the way we define our model. We will use gluon.nn.Sequential because it provides a way to rapidly add layers to our neural network.
