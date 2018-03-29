import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(0)

class ExplicitMF():
    """用ALS/SGD方法训练一个矩阵分解模型，用以预测矩阵中空缺的项目"""
    def __init__(self, ratings, n_factors=40, learning='sgd', item_fact_reg=0.0, user_fact_reg=0.0, 
                 item_bias_reg=0.0, user_bias_reg=0.0, verbose=False):
        """初始化
        Params：
        ratings : (ndarray) 用户-物品评分矩阵
        n_factors : (int) 矩阵分解模型中隐含特征的个数
        learning ：(str) 优化的方法，als或sgd
        item_fact_reg : (float) 物品隐变量正则化系数
        user_fact_reg : (float) 用户隐变量正则化系数
        item_bias_reg : (float) 物品偏差正则化系数
        user_bias_reg ：(float) 用户偏差正则化系数
        verbose : (bool) 是否输出训练进度信息
        """
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.learning = learning
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero() #获取评分矩阵中不为0的样本位置
            self.n_samples = len(self.sample_row)
        self._v = verbose
    
    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """ALS算法中的一步//type决定优化的是用户隐变量还是物品隐变量"""
        if type == 'user':
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda #np.eye(n)返回一个n*n的单位对角矩阵
            
            for u in range(latent_vectors.shape[0]):
                latent_vectors[u,:] = solve((YTY+lambdaI), ratings[u,:].dot(fixed_vecs)) #解线性方程
        
        elif type == 'item':
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(ratings.shape[1]):
                latent_vectors[i,:] = solve((XTX+lambdaI), ratings[:, i].T.dot(fixed_vecs))
        
        return latent_vectors
    
    def sgd(self):
        """SGD算法更新变量"""
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u,i)
            e = self.ratings[u,i] - prediction
            
            #更新变量
            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])
            self.user_vects[u,:] += self.learning_rate * (e * self.item_vects[i,:] - self.user_fact_reg * self.user_vects[u,:])
            self.item_vects[i,:] += self.learning_rate * (e * self.user_vects[u,:] - self.item_fact_reg * self.item_vects[i,:])
        
    def partial_train(self, n_iter):
        """部分训练，可多次调用"""
        ctr = 1
        while ctr <= n_iter:
            if ctr%10 == 0 and self._v:
                print("current iteration {}\n".format(ctr))
            if self.learning == 'als':
                self.user_vects = self.als_step(self.user_vects, self.item_vects, self.ratings, self.user_reg, type='user')
                self.item_vects = self.als_step(self.item_vects, self.user_vects, self.ratings, self.item_reg, type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1
    
    def train(self, n_iter=10, learning_rate=0.1):
        #初始化用户和物品隐变量
        """
        迭代开始前，需要对user和item的特征向量赋初值，这个初值很重要，会严重地影响到计算速度。
        一般的做法是在所有已评分的平均分附近产生一些随机数作为初值。
        这里是以0为均值的正态分布，注意到sgd的global_bias赋值为已评分项目的平均分
        """
        self.user_vects = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.item_vects = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))
        
        if self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)]) #np.where(condition)返回为true的位置信息
        
        self.partial_train(n_iter)
        
    def predict(self, u, i):
        """预测用户u对物品i的评分"""
        if self.learning == 'als':
            return self.user_vects[u,:].dot(self.item_vects[i,:].T)
        
        if self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vects[u,:].dot(self.item_vects[i,:].T)
            return prediction
    
    def predict_all(self):
        """预测整个评分矩阵"""
        predictions = np.zeros((self.user_vects.shape[0], self.item_vects.shape[0]))
        
        for u in range(self.user_vects.shape[0]):
            for i in range(self.item_vects.shape[0]):
                predictions[u,i] = self.predict(u,i)
        
        return predictions

    def get_mse(self, pred, test):
        #忽略test中为0的项
        pred = pred[test.nonzero()].flatten()
        test = test[test.nonzero()].flatten()
        return mean_squared_error(pred, test)
    
    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """在训练中实时追踪MSE值"""
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print("Iteration: {}".format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)
            
            predictions = self.predict_all()
            
            self.train_mse += [self.get_mse(predictions, ratings)]
            self.test_mse += [self.get_mse(predictions, test)]
            
            if self._v:
                print("train mse: "+ str(self.train_mse[-1]))
                print("tset mse: "+ str(self.test_mse[-1]))
            iter_diff = n_iter

    def plot_learning_curve(self, iter_array):
        plt.plot(iter_array, self.train_mse, \
                label='Training', linewidth=5)
        plt.plot(iter_array, self.test_mse, \
                label='Test', linewidth=5)


        plt.xticks(fontsize=16);
        plt.yticks(fontsize=16);
        plt.xlabel('iterations', fontsize=30);
        plt.ylabel('MSE', fontsize=30);
        plt.legend(loc='best', fontsize=20);