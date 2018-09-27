---
title: 1-1 Python Basics with Numpy
date: 2018-09-04 10:45:01
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
---

***
在iPython Notebooks中按下"SHIFT"+"ENTER"来运行相应的代码块。

***

```
import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
```

```
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
print (1.0 / x)
```

```
x_sum = np.sum(x_exp, axis = 1, keepdims = True)
```
 keepdims：是否保持矩阵的二维特性，例如使结果的形状是(4,1)而不是(4,)。

***

np.shape 和 np.reshape().

X.shape 用来获得矩阵或者向量X的形状（维度）.
X.reshape(...) 改变X的维度.
例如， 将三维的图片向量  (length,height,depth=3) 更改为一维  (length∗height∗3,1)
```
v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
```
***

求范数
x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
- ord：范数类型
默认为第二范数，即算数平方根
- axis：处理类型
axis=1表示按行向量处理，求多个行向量的范数
axis=0表示按列向量处理，求多个列向量的范数
axis=None表示矩阵范数。
- keepdims：是否保持矩阵的二维特性

```
# 正则化
# Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
x_norm = np.linalg.norm(x,axis=1,keepdims=True)
# Divide x by its norm.
x = x / x_norm
```
***
向量化
np.dot()：进行矩阵与矩阵，矩阵与向量之间的乘法。对于秩为1的数组x（即一维数组），执行对应位置相乘，然后再相加，np.dot(x)效果等同于np.sum(np.multiply(x))；对于秩不为1的二维数组，执行矩阵乘法运算。
np.multiply() 和 * 进行的是元素相乘。
乘号*：对数组执行对应位置相乘，对矩阵执行矩阵乘法运算。
```
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
dot = np.dot(x1,x2)

### VECTORIZED OUTER PRODUCT ###
outer = np.outer(x1,x2)

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
mul = np.multiply(x1,x2)

np.multiply(A,B)       #数组对应元素位置相乘，输出为array
(np.mat(A))*(np.mat(B))  #执行矩阵运算，输出为matrix
```
