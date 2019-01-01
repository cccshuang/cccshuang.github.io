---
title: 一种向量化技巧--Make Your Program Run Faster
date: 2019-01-01 19:56:23
categories:
- python
tags:
- 向量化
- numpy
mathjax: true
---

相对于编译型语言，python是解释执行的，每次在循环调用时都会先将代码解释执行，此处的消耗非常之大，而使用底层是C实现的numpy编写高度向量化的代码，可以大大提高程序执行效率。
所以我们在编写程序时，尽可能地减少for循环，可以让我们的代码变得更快。这里以k-NearestNeighbor（KNN）算法为例分享一种将代码向量化的方法。

假设我们有$N_{test}$个维度为P的测试数据$x$，$N$个维度为$P$的且具有类别标记的训练数据$x_{train}$，根目标是据训练数据使用KNN将每一个测试数据分别划分到合适的类别。那么怎么实现呢？
首先想到的一种方法是对于每一个测试数据$x[i]$，我们分别计算$x[i]$与$N$个训练数据$x_{train}$的距离，这样对于$x[i]$可以得到长度为$N$的距离数组$dis$，将其从小到大排序，取距离较小的前$k$个训练数据，然后我们根据这$k$个数据中最多的类别来确定$x[i]$的类别。实现如下：

```python
import numpy as np
import scipy.stats

def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.
        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        dis = np.linalg.norm(x_train - x[i, :], ord=2, axis=1, keepdims=True).flatten() # 耗时
        top_k = y_train[np.argsort(dis)[:k]]
        y[i] = scipy.stats.mode(top_k).mode

    return y
```
之前我们提到在编写程序时，尽可能地减少for循环，上面程序中出现了一层循环，那么我们有没有办法把这一层循环也去掉呢，从而将我们的程序完全向量化。当然是可以的，仅仅将测试数据和训练数据的维度进行一个简单的变换即可。测试数据$x$的原来维度为$(N_{test},P)$，我们将其扩展一维变成$(N_{test},1,P)$，而对于训练数据$x_{train}$原来维度为$(N,P)$，我们也将其扩展一维变成$(1,N,P)$。这样当我们计算测试数据与训练数据距离的时候，由于numpy的广播机制，$x_{train} - x$的维度就变成了$(N_{test},N,P)$，对其第3个维度求和，得到距离矩阵$dis$，维度为$(N_{test},N,1)$，代表每一个测试数据点与$N$个训练数据的距离，对于每个测试数据点根据其与训练数据的距离进行排序，取前$k$个，然后我们根据这$k$个数据中最多的类别来确定$x[i]$的类别。实现如下：

```python
import numpy as np
import scipy.stats

def knn(x, x_train, y_train, k):
    x_ = x.reshape((x.shape[0], 1, x.shape[1]))
    x_train_ = x_train.reshape((1, x_train.shape[0],x_train.shape[1]))
    dis = np.sum(np.square(x_train_ - x_), axis = 2)
    top_k = y_train[np.argsort(dis, axis=1)[:, :k]]
    y = scipy.stats.mode(top_k, axis=1).mode  

    return y
```
使用一些数据进行测试，在我的计算机上，完全向量化的第二种方法相较于第一种含有for循环的实现，达到了4倍的加速比，第二种方法的实现也体现了一种空间换时间的思想。由此可见向量化对于提高程序运行效率是非常有效的一种方法！