---
title: 2-1 Initialization
date: 2018-09-18 14:41:03
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
mathjax: true
---


#### 算法步骤
我们以一个三层神经网络为例，介绍三种不同的超参数初始化方法对模型的影响，分别是：
- Zeros initialization -- setting initialization = "zeros" in the input argument.
- Random initialization -- setting initialization = "random" in the input argument. This initializes the weights to large random values.
- He initialization -- setting initialization = "he" in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015.

下面首先是我们的三层网络模型，通过传入不同的 initialization 值来调用不同的初始化方法。

```
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):

    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```
下面对这三种初始化方法进行讨论。
对于 Zeros initialization ，表现非常糟糕，损失几乎在训练过程中不下降，表现不比随机猜测要好。这是为什么呢？因为将超参数初始化为0，由于对称性，导致在训练过程中每层的神经元都在做相同的计算，这样相当于每层只有一个神经元，从而神经网络的性能大大降低。因此W[l]需要通过初始化打破这种对称性（break symmetry）。
对于 Random initialization ，虽然进行了初始化，打破了上面所说的对称性，但是初始化的值都较大的话，会使得初始的损失较大，这是因为当超参数的值变大后，最后一层中 sigmoid 的输出对于一些样本会更加接近于 0 或 1 ，当样本预测错误后对于损失函数，例如  log(a[3])=log(0)log⁡(a[3])=log⁡(0) ，损失就变成了无穷大。这种情况减缓了神经网络的收敛。
对于 He initialization ， 对超参数随机初始化后，乘以一个缩放因子 sqrt(2./layers_dims[l-1]).) ，适合使用于带有 RELU 激活函数的层。其实现如下：
```
def initialize_parameters_he(layers_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2.0 / layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters
```
