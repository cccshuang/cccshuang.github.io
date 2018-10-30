---
title: '4-3 Keras Tutorial'
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
mathjax: true
---

np.expand_dims:用于扩展数组的形状
```
>>> x = np.array([1,2])
>>> x.shape
(2,)
>>> y = np.expand_dims(x, axis=0)
>>> y
array([[1, 2]])
>>> y.shape
(1, 2)
>>> y = np.expand_dims(x, axis=1)  # Equivalent to x[:,newaxis]
>>> y
array([[1],
       [2]])
>>> y.shape
(2, 1)
```