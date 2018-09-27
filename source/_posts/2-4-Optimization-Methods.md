---
title: 2-4 Optimization Methods
date: 2018-09-26 09:42:28
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
mathjax: true
---

np.sqrt(x) 计算数组各元素的平方根 
np.square(x) 计算数组各元素的平方

```
"""
打印结果:
加法运算: 5+8=13
减法运算: 5-8=-3
乘法运算: 5*8=40
除法运算: 5/8=0.625
"""
#函数或lambda表达式作为参数传参
def calculate(x, y, func):
    return func(x, y)
#加法
def add(x, y):
    return x + y
#减法
def sub(x, y):
    return x - y
a,b = 5,8
add_ret = calculate(a, b, add)  #加法
sub_ret = calculate(a, b, sub)  #减法
mul_ret = calculate(a, b, lambda a,b : a*b)  #乘法
dev_ret = calculate(a, b, lambda a,b : a/b)  #除法
 
print('加法运算: {}+{}={}'.format(a, b, add_ret))
print('减法运算: {}-{}={}'.format(a, b, sub_ret))
print('乘法运算: {}*{}={}'.format(a, b, mul_ret))
print('除法运算: {}/{}={}'.format(a, b, dev_ret))
```


本文介绍几种优化算法，可以加速学习，甚至提升学习效果。
#### 梯度下降法
梯度下降法是（Gradient Descent (GD)）机器学习中最简单的优化算法。
将所有 $m$ 个算例用于一步梯度下降时，称为 Batch Gradient Descent。当训练集较大时，收敛速度比较慢。
将一个算例用于一步梯度下降时，称为 Stochastic（随机的） Gradient Descent (SGD)。当训练集较大时，收敛速度快，但是会在聚集点处震荡。
将$n$个算例用于一步梯度下降时（$1<n<m$），称为 Mini-Batch Gradient descent。

#### Mini-Batch 梯度下降法
首先建立训练集(X, Y)的 mini-batches。分为两步，第一步是将(X, Y)随机打乱，这里要保证 X 和 Y 打乱后也要一一对应；第二步是进行划分，划分的大小一般取2的指数，如 16, 32, 64, 128，注意，最后一个 mini-batch 也许不够划分大小，不过这对训练没有影响。

```

def random_mini_batches(X, Y, mini_batch_size = 64:
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
```

#### Momentum 梯度下降法
mini-batch梯度下降法仅使用部分训练集来更新参数，因此会产生一些偏差，在收敛的过程中震荡，我们使用 Momentum 通过对梯度求指数加权平均来平滑梯度，从而避免震荡。不仅mini-batch梯度下降法，也可以把 Momentum 加到batch gradient descent 或者 stochastic gradient descent中去。
首先我们需要使用参数 $v$ 来代表过去的梯度，由此来求指数加权平均。对其进行初始化为0，$v$ 与 参数 $dW$ 和 $db$ 大小相同。
```

def initialize_velocity(parameters):

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1]))
        v['db' + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape[0], parameters['b' + str(l+1)].shape[1]))
        
    return v
```
然后更新参数。

$$ \begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\\\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases}$$

$$\begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\\\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
\end{cases}$$

实现如下：
```

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
        
    return parameters, v
```
如果 $\beta = 0$，那么就成了普通的梯度下降法。$\beta$ 越大，越平滑，因为把过去的梯度更多地考虑进去了，但是过大也会使梯度过度平滑。
常用的 $\beta$ 值取 0.8 到 0.999。如果感觉没有必要调这个参数值, 一般取 $\beta = 0.9$。

#### Adam 梯度下降法
Adam 是训练神经网络最有效的优化算法之一。它同时结合了RMSProp 和 Momentum。
首先计算过去梯度的指数加权平均值，存在参数 $v$ 中，并进行偏差修正，得到 $v^{corrected}$。
然后计算过去梯度平方的指数加权平均值，存在参数 $s$ 中，并进行偏差修正，得到 $s^{corrected}$。
最后结合前两步更新参数。

$$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\\\
v_{dW^{[l]}}^{corrected} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\\\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\\\
s_{dW^{[l]}}^{corrected} = \frac{s_{dW^{[l]}}}{1 - (\beta_1)^t} \\\\
W^{[l]} = W^{[l]} - \alpha \frac{v_{dW^{[l]}}^{corrected}}{\sqrt{s_{dW^{[l]}}^{corrected}} + \varepsilon}
\end{cases}$$

下面进行实现，首先对 $v,s$ 进行初始化。
```

def initialize_adam(parameters) :
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L): 
        v['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1]))
        v['db' + str(l + 1)] = np.zeros((parameters['b' + str(l+1)].shape[0], parameters['b' + str(l+1)].shape[1]))
        s['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1]))
        s['db' + str(l + 1)] = np.zeros((parameters['b' + str(l+1)].shape[0], parameters['b' + str(l+1)].shape[1]))
 
    return v, s
```
对参数进行更新。
```

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta1 ** t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - beta1 ** t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * (grads['dW' + str(l + 1)] ** 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * (grads['db' + str(l + 1)] ** 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".

        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta2 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v_corrected['dW' + str(l + 1)] / (np.sqrt(s_corrected['dW' + str(l + 1)]) + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected['db' + str(l + 1)] / (np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon)

    return parameters, v, s
```
Adam 中，超参数 learning_rate 是需要调的，一般设置 beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8。

#### 总结
最后，我们怎么在模型中使用这些优化算法呢？

```
def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
```
调用模型。
```
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")
```
Momentum 通常有所帮助，如果学习率较低且数据集过于简单的话，其影响几乎可以忽略不计。 
另一方面，Adam 明显优于mini-batch 梯度下降和Momentum。
如果在数据集上运行更多时间，则所有这三种方法都将产生非常好的结果。 但是，Adam 收敛得更快。
