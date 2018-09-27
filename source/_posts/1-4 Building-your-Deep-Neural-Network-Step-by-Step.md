---
title: '1-4 Building your Deep Neural Network: Step by Step'
date: 2018-09-16 14:55:23
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
mathjax: true
---

注意np.zeros(shape)中的shape是有括号的形式。

```
W1 = np.random.randn(n_h,n_x)*0.01
b1 = np.zeros((n_h, 1))
```
python中，//为整除符号，取整数部分。

```
>>> 1/2
0.5
>>> 1//2
0
```

#### 算法步骤

1. 初始化参数
我们使用layer_dims来存储每一层单元的数量。 如layer_dims = [2,4,1] : 第一个是输入层有两个单元，第二个是隐藏层有4个单元，第三个是输出层有1个单元，因此W1大小为 (4,2)， b1 为 (4,1)，W2 为 (1,4) ， b2 为 (1,1)。 

```

def initialize_parameters_deep(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network，包括输入层，隐藏层和输出层总共L-1个

    for l in range(1, L):
        #Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        #bl -- bias vector of shape (layer_dims[l], 1)
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters
```

2. 前向传播模块
首先实现线性前向传播

```
def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    """
    
    Z = np.dot(W, A) + b

    cache = (A, W, b)
    
    return Z, cache
```

加上激活函数

```
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    #linear_cache = (A_prev, W, b), activation_cache = Z
    cache = (linear_cache, activation_cache)

    return A, cache
```

实现 L 层神经网络，堆叠使用 RELU 的 linear_activation_forward 函数 L−1 次, 最后堆叠一个使用 SIGMOID 的 linear_activation_forward 函数。

```

def L_model_forward(X, parameters):
    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network，不包括输入层，仅包括隐藏层和输出层。

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # cache = (linear_cache, activation_cache) = ((A_prev, W, b), Z)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
     
    return AL, caches
```

3. 计算损失函数

```
def compute_cost(AL, Y):
   
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m 
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost
```

4. 反向传播模块
对于层$l$的线性部分 $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$。
假设已经知道 $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$，计算：
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} $$

$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l][i]}$$

$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} $$

其中，$dZ^{[l]}: (n^{[l]},m)$、$A^{[l-1]}: (n^{[l-1]},m)$、$W^{[l]}: (n^{[l]},n^{[l-1]})$。


```

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m

    return dA_prev, dW, db
```
下面求$dZ^{[l]}$，对于激活函数部分
$$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$$


```

def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- post-activation gradient for current layer l 

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    """
    # cache = (linear_cache, activation_cache) = ((A_prev, W, b), Z)
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
```

堆叠L层反向传播。首先初始化反向传播，我们知道在第 L 层，$A^{[L]} = \sigma(Z^{[L]})$，要计算$Z^{[L]}$关于激活函数的导数，首先要计算出$A^{[l]}$关于损失函数的导数$ dA^{[L]} = \frac{\partial \mathcal{L}}{\partial A^{[L]}}$ ，以作为def linear_activation_backward(dA, cache, activation)的初值，之后的$ dA^{[L-1]} \dots dA^{[1]}$均可由此函数递推出来。
$ dA^{[L]} $计算方法如下:
`dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) `

```

def L_model_backward(AL, Y, caches):
    """
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[L-1], activation = "sigmoid")
    
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 2)], caches[l], activation = "relu")

    return grads
```

5. 更新参数

```

def update_parameters(parameters, grads, learning_rate):
   
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters
```

6. 预测函数
```
def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
```





#### 附录1：SIGMOID和RELU函数实现

```
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache
```
#### 附录2：SIGMOID和RELU反向传播函数实现

relu_backward中dZ[Z <= 0] = 0这一步不是很懂，不应该是大于等于0时导数为1吗？

```
def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
```