---
title: 1-2 Logistic Regression with a Neural Network mindset
date: 2018-09-04 21:00:54
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
---

#### 算法步骤
1. 首先实现函数initialize_with_zeros(dim)对参数w和b进行初始化
```
w = np.zeros((dim, 1))
b = 0
```
2. 实现前向和后向传播算法 propagate(w, b, X, Y)，在此函数里计算损失cost，并求出w和b的梯度dw和db。
```
A = sigmoid( np.dot(w.T, X) + b )           # compute activation
cost = -  np.sum( Y * np.log(A) + (1-Y) * np.log(1-A))  / m   # compute cost

dw = np.dot(X,(A-Y).T) / m
db = np.sum(A-Y) / m
```
3. 实现优化函数optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False)进行梯度下降。
```
 costs = []
 for i in range(num_iterations):
    # Cost and gradient calculation (≈ 1-4 lines of code)
    grads, cost = propagate(w, b, X, Y)

    # Retrieve derivatives from grads
    dw = grads["dw"]
    db = grads["db"]
        
    # update rule 
    w = w - learning_rate*dw
    b = b - learning_rate*db
    if i % 100 == 0:
        costs.append(cost)
```
最后返回的是 params, grads 和 costs。
```
params = {"w": w,
            "b": b}
    
grads = {"dw": dw,
            "db": db}
```
4. 实现预测函数 predict(w, b, X)，返回预测结果 Y_prediction。
```
A = sigmoid( np.dot(w.T, X) + b )
for i in range(A.shape[1]):
    if A[0][i] <= 0.5:
        Y_prediction[0][i] = 0
    else:
        Y_prediction[0][i] = 1
```
5. 将上述几步结合，实现最后的模型 model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False)。
```
w, b = initialize_with_zeros(X_train.shape[0])
params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
w = params['w']
b = params['b']
Y_prediction_test = predict(w, b, X_test)
Y_prediction_train = predict(w, b, X_train)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
```

#### numpy部分函数
np.squeeze() 从数组的形状中删除单维条目，即把shape中为1的维度去掉
```
import numpy as np

x = np.array([[[0], [1], [2]]])
print(x.shape)  # (1, 3, 1)

x1 = np.squeeze(x)  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
print(x1)  # [0 1 2]
print(x1.shape)  # (3,)
```

将矩阵 X : (a,b,c,d) 变换为矩阵 X_flatten ： (b ∗ c ∗ d, a) 时，使用
```
X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X，参数为-1时，reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值。
```