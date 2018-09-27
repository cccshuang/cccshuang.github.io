---
title: '1-5 Deep Neural Network for Image Classification: Application'
date: 2018-09-17 14:00:11
categories:
- 深度学习习题
tags:
- DeepLearning
- 习题
---

将4 Building your Deep Neural Network: Step by Step中实现的神经网络应用到图像识别中，这里的任务是识别一张图中是否有猫。


##### 数据预处理
数据的一些信息：
Number of training examples: 209
Number of testing examples: 50
Each image is of size: (64, 64, 3)
train_x_orig shape: (209, 64, 64, 3)
train_y shape: (1, 209)
test_x_orig shape: (50, 64, 64, 3)
test_y shape: (1, 50)

```
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0] # 训练集中图片的数量
num_px = train_x_orig.shape[1] # 图片的维度
m_test = test_x_orig.shape[0] #测试集中图片的数量

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
```
预处理后：
train_x's shape: (12288, 209)
test_x's shape: (12288, 50)


##### L层的神经网络

```
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
```

```

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

如何进行调用上面的模型呢？
```
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
```
进行预测
```
pred_test = predict(test_x, test_y, parameters)
```

tips:如何找到误分类的图片呢？
将预测结果pred_test和test_y相加，如果等于1的位置的对应图片就是误分类的图片。

|pred_test  | test_y | pred_test+test_y | 结果 |
| :------:  | :------:  | :------:  | :------:  |
| 0 | 0 | 0 | 分类正确|
| 0 | 1 | 1 | 误分类  |
| 1 | 0 | 1 | 误分类  |
| 1 | 1 | 2 | 分类正确|

```
a = pred_test + test_y
mislabeled_indices = np.asarray(np.where(a == 1))
```