---
title: Machine Learning Assignment 1
date: 2018-12-03 11:10:34
categories:
- Machine Learning Assignment
tags:
- Machine Learning
- 习题
mathjax: True
---

1. Visualize very high dimensional data in 2D or 3D space.
t-SNE聚类

2. Represent image as a well chosen 64 bits integer, so that similar images will be
represented as integers with small hamming distance
[参考](http://www.ruanyifeng.com/blog/2011/07/principle_of_similar_image_search.html)
Google、Baidu 等搜索引擎相继推出了以图搜图的功能，测试了下效果还不错~ 那这种技术的原理是什么呢？计算机怎么知道两张图片相似呢？
根据Neal Krawetz博士的解释，原理非常简单易懂。我们可以用一个快速算法，就达到基本的效果。
这里的关键技术叫做”感知哈希算法”（Perceptual hash algorithm），它的作用是对每张图片生成一个”指纹”（fingerprint）字符串，然后比较不同图片的指纹。结果越接近，就说明图片越相似。
下面是一个最简单的实现：
第一步，缩小尺寸。
将图片缩小到8x8的尺寸，总共64个像素。这一步的作用是去除图片的细节，只保留结构、明暗等基本信息，摒弃不同尺寸、比例带来的图片差异。
第二步，简化色彩。
将缩小后的图片，转为64级灰度。也就是说，所有像素点总共只有64种颜色。
第三步，计算平均值。
计算所有64个像素的灰度平均值。
第四步，比较像素的灰度。
将每个像素的灰度，与平均值进行比较。大于或等于平均值，记为1；小于平均值，记为0。
第五步，计算哈希值。
将上一步的比较结果，组合在一起，就构成了一个64位的整数，这就是这张图片的指纹。组合的次序并不重要，只要保证所有图片都采用同样次序就行了。
= = 8f373714acfcf4d0
得到指纹以后，就可以对比不同的图片，看看64位中有多少位是不一样的。在理论上，这等同于计算”汉明距离”（Hamming distance）。如果不相同的数据位不超过5，就说明两张图片很相似；如果大于10，就说明这是两张不同的图片。
这种算法的优点是简单快速，不受图片大小缩放的影响，缺点是图片的内容不能变更。如果在图片上加几个文字，它就认不出来了。所以，它的最佳用途是根据缩略图，找出原图。

3. [三门问题与贝叶斯理论](https://blog.csdn.net/zjuPeco/article/details/76850866)
如果参赛者换的话，那么参赛者会在最初选择是错误的时候获得汽车；如果参赛者不换的话，那么参赛者会在最初选择是正确的时候获得汽车。 
前者是$\frac{2}{3}$的概率，后者是$\frac{1}{3}$的概率

4. ipython在调试过程中，如果代码发生更新，怎么实现ipython中引用的模块也自动更新呢。
`%load_ext autoreload `
`%autoreload 2`

5. `i,j,ham_test = np.loadtxt('ham_test.txt',dtype=int).T` 可以通过指定dtype来指定读入的类型。
6. `ordered_r = list(reversed(np.argsort(ratio)))`此代码中，reversed返回的是一个迭代器，需要使用list获取所有值，np.argsort是返回数组排序后的元素对应的下标。`words[ordered_r[0:10]][:,0]`通过排序后的数组下标获得原数组中前k大的元素。

7. 当计算$p(x|y) = \prod_{i} p(x_i|y)$时，浮点数相乘可能会下溢，所以我们取其对数，转换为$log p(x|y) = \sum_{i} log p(x_i|y)$。