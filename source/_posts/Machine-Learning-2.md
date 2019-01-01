---
title: Machine Learning 2
date: 2018-12-12 09:38:03
tags:
---
```
>>> numpy.sign(x) #  returns -1 if x < 0, 0 if x==0, 1 if x > 0
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[2], [3], [4]])
>>> np.vstack((a,b)) # Equivalent to np.concatenate(tup, axis=0)
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```

```
numpy.random.random(size=None) # Return random floats in the half-open interval [0.0, 1.0).
#To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
>>> 5 * np.random.random_sample((3, 2)) - 5 # Three-by-two array of random numbers from [-5, 0):
array([[-3.99149989, -0.52338984],
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
```