---
title: 2-3 Gradient Checking
date: 2018-09-20 15:58:13
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
mathjax: true
---

np.linalg.norm(x, ord=None, axis=None, keepdims=False) 求向量（矩阵）范数。

|参数  | 说明 | 计算方法 | 
| :------:  | :------:  | :------:  |
| 默认 | 二范数：$l_2$ | $\sqrt{x_{1}^{2}+x_{2}^{2}+\cdots+x_{n}^{2}}$ | 
| ord = 2 | 二范数：$l_2$  | $\sqrt{x_{1}^{2}+x_{2}^{2}+\cdots+x_{n}^{2}}$ | 
| ord = 1 | 一范数：$l_1$  | $\lvert x_{1} \rvert + \cdots + \lvert x_{n} \rvert $ | 
| ord=np.inf | 无穷范数 $l_{\infty}$ | $\max({\lvert x_{i}) \rvert}$ | 


```
>>> x = np.array([3, 4])
>>> np.linalg.norm(x)
5.
>>> np.linalg.norm(x, ord=2)
5.
>>> np.linalg.norm(x, ord=1)
7.
>>> np.linalg.norm(x, ord=np.inf)
4
```

#### 算法步骤
梯度检查（Gradient Checking）用于检查实现的反向传播算法是否正确。利用前向传播函数，通过公式
$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$
来计算导数的近似值，与反向传播计算得到的导数进行对比，来判断反向传播是否正确。
具体步骤如下：
- 首先计算导数的近似值
    1. $\theta^{+} = \theta + \varepsilon$
    2. $\theta^{-} = \theta - \varepsilon$
    3. $J^{+} = J(\theta^{+})$
    4. $J^{-} = J(\theta^{-})$
    5. $gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}$
- 然后计算反向传播算法得到的导数grad
- 最后通过下面公式来对比两个值的差别大小，若很小，如小于$10^{-7}$，便认为反向传播算法的实现是正确的。
$$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} $$

当有很多个参数时，分别对每个进行梯度检查。算法步骤如下：
For each i in num_parameters:
- To compute `J_plus[i]`:
    1. Set $\theta^{+}$ to `np.copy(parameters_values)`
    2. Set $\theta^{+}_i$ to $\theta^{+}_i + \varepsilon$
    3. Calculate $J^{+}_i$ using to `forward_propagation_n(x, y, vector_to_dictionary(`$\theta^{+}$ `))`.     
- To compute `J_minus[i]`: do the same thing with $\theta^{-}$
- Compute $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$

```

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        theta_plus =  np.copy(parameters_values)                                     # Step 1
        theta_plus[i] = theta_plus[i] +  epsilon                            # Step 2
        J_plus[i], _ =  forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))                                  # Step 3
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        theta_minus =  np.copy(parameters_values)                                    # Step 1
        theta_minus[i] = theta_minus[i] - epsilon                          # Step 2        
        J_minus[i], _ =  forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))                                   # Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                              # Step 1'
    denumerator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                          # Step 2'
    difference =   numerator / denumerator                                               # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
```

注意，进行梯度检查是很耗时的，仅仅是用来检查反向传播的实现是否正确，所以我们在训练过程中不能加进去。而且梯度检查不能用在使用dropout的神经网络中，进行梯度检查时要关闭dropout。